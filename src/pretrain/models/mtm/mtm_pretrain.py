import multiprocessing as mp

import hydra
import numpy as np
import torch
import torch.nn as nn
from rich.console import Console
from rich.table import Table
from torch.nn import PoissonNLLLoss
from torch.utils.data import DataLoader
from torch_brain.batching import collate
from torch_brain.samplers import RandomFixedWindowSampler, SequentialFixedWindowSampler
from torch_brain.transforms import Compose

from core.samplers import DistributedSamplerWrapper, SessionBatchSampler
from core.trainer import BaseTrainer
from core.utils.util import log_param_breakdown, move_to_device
from ibl_bwb_eval.metrics import BPS, PoissonD2Score
from pretrain.datasets import IBLBrainWideBenchMaskModelingSpikes

from .masker import MtMMasker
from .mtm import MtM


class MtMPretrain(BaseTrainer):
    """Masked spike modelling for MtM, whose masks carry a mode token."""

    def setup(self, ckpt: dict | None):
        self.logger.info(f"Trainer: {self.__class__}")
        self.reset_best_tracking(minimize=True)

        # task
        self.logger.info("Task: Mask Modeling (BERT style)")
        self.loss_fn = PoissonNLLLoss(log_input=True, full=True, eps=1e-9, reduction="none")

        # data loaders
        self.setup_train_loader()
        self.setup_val_loader()

        # model
        self.setup_model(ckpt)
        self.link_model(self.model, ckpt)

        # move model to device after linking, in case new parameters were created
        self.model.to(self.device)
        self.model = self.make_ddp(self.model)

        # optimizer and scheduler
        self.setup_optimizers(ckpt)

    def setup_train_loader(self):
        self.train_dataset = IBLBrainWideBenchMaskModelingSpikes(
            root=self.cfg.data_root,
            recording_ids=self.cfg.recording_ids,
            split="train",
        )

        self.train_batch_sampler = DistributedSamplerWrapper(
            SessionBatchSampler(
                RandomFixedWindowSampler(
                    sampling_intervals=self.train_dataset.get_trial_intervals(),
                    window_length=self.train_dataset.context_window,
                    generator=torch.Generator().manual_seed(self.cfg.seed),
                    drop_short=True,
                ),
                batch_size=self.cfg.batch_size // self.world_size,
                shuffle_batches=True,
                generator=torch.Generator().manual_seed(self.cfg.seed),
            )
        )

        has_workers = self.cfg.num_workers > 0
        self.train_loader = DataLoader(
            self.train_dataset,
            batch_sampler=self.train_batch_sampler,
            collate_fn=collate,
            num_workers=self.cfg.num_workers,
            pin_memory=self.cfg.pin_memory,
            prefetch_factor=self.cfg.prefetch_factor if has_workers else None,
            persistent_workers=self.cfg.persistent_workers if has_workers else False,
            multiprocessing_context=mp.get_context("fork") if has_workers else None,
        )

        self.logger.info(
            f"Train dataset: {len(self.train_dataset.get_session_ids())} sessions, {len(self.train_dataset.get_unit_ids())} units"
        )

        self.logger.info(
            f"Training on {len(self.train_loader)} batches, total {len(self.train_loader) * self.cfg.batch_size} samples"
        )

    def setup_val_loader(self):
        self.val_dataset = IBLBrainWideBenchMaskModelingSpikes(
            root=self.cfg.data_root,
            recording_ids=self.cfg.recording_ids,
            split="val",
        )

        self.val_batch_sampler = DistributedSamplerWrapper(
            SessionBatchSampler(
                SequentialFixedWindowSampler(
                    sampling_intervals=self.val_dataset.get_trial_intervals(),
                    window_length=self.val_dataset.context_window,
                    drop_short=True,
                ),
                batch_size=self.cfg.batch_size // self.world_size,
                drop_last=False,
            )
        )

        has_workers = self.cfg.num_workers > 0
        self.val_loader = DataLoader(
            self.val_dataset,
            batch_sampler=self.val_batch_sampler,
            collate_fn=collate,
            num_workers=self.cfg.num_workers,
            pin_memory=self.cfg.pin_memory,
            prefetch_factor=self.cfg.prefetch_factor if has_workers else None,
            persistent_workers=self.cfg.persistent_workers if has_workers else False,
            multiprocessing_context=mp.get_context("fork") if has_workers else None,
        )
        self.logger.info(
            f"Val dataset: {len(self.val_dataset.get_session_ids())} sessions, {len(self.val_dataset.get_unit_ids())} units"
        )
        self.logger.info(
            f"Validating on {len(self.val_loader)} batches, total {len(self.val_loader) * self.cfg.batch_size} samples"
        )

    def setup_model(self, ckpt: dict | None = None):
        self.model = hydra.utils.instantiate(self.cfg.model)
        if not isinstance(self.model, MtM):
            raise TypeError(
                f"{type(self).__name__} requires a MtM model, got {type(self.model).__name__}"
            )
        self.masker = hydra.utils.instantiate(self.cfg.masker)
        if not isinstance(self.masker, MtMMasker):
            raise TypeError(
                f"{type(self).__name__} requires a MtMMasker masker, got {type(self.masker).__name__}"
            )
        self.add_checkpoint_items(model=self.model, masker=self.masker)

        self.logger.info(f"Precision: {self.precision}")
        self.logger.info(f"Model: {self.model.__class__}")
        self.logger.info(f"Masker: {self.masker.__class__}")
        self.model.train()

    def link_model(self, model: nn.Module, ckpt: dict | None = None):
        # attach model input_fn to end of dataset transform pipelines
        for split in ["train", "val"]:
            model_transforms = hydra.utils.instantiate(self.cfg.get(f"{split}_transforms", []))
            self.logger.info(f"Model transforms ({split}): {model_transforms}")
            dataset = getattr(self, f"{split}_loader").dataset

            if dataset.transform is None:
                dataset.transform = Compose([*model_transforms, model.input_fn])
            else:
                dataset.transform = Compose([dataset.transform, *model_transforms, model.input_fn])

        # link datasets to model
        model.link_datasets(self.train_loader.dataset, self.val_loader.dataset)

        log_param_breakdown(self.logger, self.model, ("in_stitcher", "out_stitcher"))

    def setup_optimizers(self, ckpt: dict | None = None):
        if self.model is None:
            raise ValueError("Trying to configure optimizers before the model is setup")
        if self.train_loader is None:
            raise ValueError("Trying to configure optimizers before the train data loader is setup")

        grouped_parameters = self.get_param_groups()

        max_lr = self.cfg.base_lr * np.sqrt(self.cfg.batch_size)
        self.optimizer = torch.optim.AdamW(
            grouped_parameters,
            lr=max_lr,
        )
        self.scheduler = torch.optim.lr_scheduler.OneCycleLR(
            self.optimizer,
            max_lr=max_lr,
            epochs=self.cfg.num_epochs,
            steps_per_epoch=len(self.train_loader),
            pct_start=self.cfg.pct_start,
            anneal_strategy="cos",
            div_factor=self.cfg.div_factor,
            final_div_factor=1e4,
        )
        self.add_checkpoint_items(optimizer=self.optimizer, scheduler=self.scheduler)

        return self.optimizer, self.scheduler

    def train_epoch(self):
        self.model.train()
        loader = self.train_loader
        if self.cfg.log_train_step:
            loader = self.logger.get_pbar(loader, prefix="train")

        for X in loader:
            X = move_to_device(X, device=self.device)
            self.optimizer.zero_grad()
            with torch.autocast(device_type="cuda", dtype=self.precision.dtype):
                spikes = X["model_inputs"]["spikes"]
                target_spikes = spikes.clone()
                regions = X["regions"]

                masked_spikes, mask, mask_mode = self.masker(spikes, regions)
                X["model_inputs"]["spikes"] = masked_spikes

                pred_rates = self.predict(X, mask_mode=mask_mode)

                loss = self.loss(pred_rates, target_spikes, mask)

            loss.backward()
            self.clip_and_log_grad_norm()
            self.optimizer.step()
            self.scheduler.step()
            self.train_step += 1

            if self.cfg.log_train_step:
                self.logger.log("train/loss", loss.item(), pbar=True)
                self.logger.log(f"train/loss_{mask_mode}", loss.item(), pbar=False)
                self.logger.log("train/lr", self.scheduler.get_last_lr()[0], pbar=False)
                self.push_logs()

    def predict(self, X, mask_mode=None):
        return self.model(**X["model_inputs"], mask_mode=mask_mode)

    def loss(self, pred_rates, target_spikes, mask):
        loss = self.loss_fn(pred_rates, target_spikes)
        if mask.sum() == 0:
            return loss.sum() * 0.0
        return loss[mask].mean()

    @torch.inference_mode()
    def val_epoch(self):
        self.model.eval()
        mask_types = self.masker.mask_types

        # Pre-compute one deterministic mask per (session, mode) before the loop.
        session_masks = self._precompute_val_masks(mask_types)

        loss_sum, loss_count = 0.0, 0
        mode_loss_sum: dict[str, float] = {}
        mode_loss_count: dict[str, int] = {}
        # Lazily built on first encounter of each (mode, session) pair, which is what
        # keeps this trainer single-GPU. See _build_mode_metrics.
        mode_session_diag: dict[tuple, dict] = {}
        updated_keys: set[tuple] = set()

        for batch_idx, X in enumerate(self.val_loader):
            X = move_to_device(X, device=self.device)
            with torch.autocast(device_type="cuda", dtype=self.precision.dtype):
                spikes = X["model_inputs"]["spikes"]
                target_spikes = spikes.clone()
                session_idx = int(X["model_inputs"]["session_tokens"].unique().item())

                # Cycle through modes deterministically; apply the pre-computed
                # mask for this (session, mode) pair.
                forced_mode = mask_types[batch_idx % len(mask_types)]
                spec = session_masks[(session_idx, forced_mode)]
                B = spikes.shape[0]
                mask = spec["is_mask"].unsqueeze(0).expand(B, -1, -1)
                input_zero = spec["input_zero_mask"].unsqueeze(0).expand(B, -1, -1)

                masked_spikes = spikes.clone()
                masked_spikes[input_zero] = 0.0
                X["model_inputs"]["spikes"] = masked_spikes

                pred_rates = self.predict(X, mask_mode=forced_mode)

                loss = self.loss(pred_rates, target_spikes, mask)
                mask_count = int(mask.sum().item())
                loss_sum += float(loss.item()) * mask_count
                loss_count += mask_count
                mode_loss_sum[forced_mode] = (
                    mode_loss_sum.get(forced_mode, 0.0) + float(loss.item()) * mask_count
                )
                mode_loss_count[forced_mode] = mode_loss_count.get(forced_mode, 0) + mask_count

                key = (forced_mode, session_idx)
                if key not in mode_session_diag:
                    mode_session_diag[key] = self._build_mode_metrics(forced_mode, spec["is_mask"])
                did_update = self._update_mode_metrics(
                    mode_session_diag[key],
                    forced_mode,
                    spec["is_mask"],
                    pred_rates,
                    target_spikes,
                )
                if did_update:
                    updated_keys.add(key)

        val_loss = self.reduce_mean(loss_sum, loss_count)
        metrics = {
            "avg": val_loss,
            "loss": val_loss,
            "epoch": self.epoch,
        }
        for mode in mask_types:
            metrics[f"loss_{mode}"] = self.reduce_mean(
                mode_loss_sum.get(mode, 0.0), mode_loss_count.get(mode, 0)
            )

        metrics.update(self._aggregate_mode_metrics(mode_session_diag, mask_types, updated_keys))

        improved = self.save_if_best(val_loss, metrics=metrics)
        early_stop = self.step_patience(improved)

        val_metrics = {f"val/{k}": v for k, v in metrics.items()}
        self.logger.log_dict(val_metrics)
        self.push_logs()
        self.logger.update_epoch_postfix(self.get_best_pbar_metrics())
        self.return_value = val_metrics
        return early_stop

    def get_best_pbar_metrics(self) -> dict:
        if not self.best_metrics:
            return {}
        out = {}
        for k, v in self.best_metrics.items():
            if "epoch" in k:
                continue
            bare = k.replace("best/val/", "")
            # drop per-session entries (e.g. "bps/<session_uuid>") to keep the
            # postfix readable; only show top-level keys and "*/avg" aggregates.
            if "/" in bare and not bare.endswith("/avg"):
                continue
            out[bare] = v
        return out

    def log_training_results(self):
        table = Table(title="Final Best Metrics", show_header=True)
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="green")
        for k, v in sorted(self.best_metrics.items(), key=lambda item: item[0]):
            table.add_row(k, f"{v:.4f}" if isinstance(v, float) else str(v))
        Console().print(table)

    def _precompute_val_masks(self, mask_types):
        """Run the masker once per (session, mode) on a B=1 slice, seeded by
        session index, so every batch of the same session sees identical masked
        positions throughout the epoch.

        Returns a dict keyed by (session_idx, mode) with:
            is_mask        (T, N) bool: which positions to predict / evaluate
            input_zero_mask (T, N) bool: which positions to zero in the input
                                          (superset of is_mask for intra_region,
                                          which also zeros non-target-region neurons)
        """
        session_first = {}
        for X in self.val_loader:
            X = move_to_device(X, device=self.device)
            session_idx = int(X["model_inputs"]["session_tokens"].unique().item())
            if session_idx not in session_first:
                session_first[session_idx] = X

        specs = {}
        for session_idx, X in session_first.items():
            spikes = X["model_inputs"]["spikes"][:1]  # (1, T, N)
            regions = X["regions"][:1] if X["regions"] is not None else None
            ones = torch.ones_like(spikes)
            for mode_idx, mode in enumerate(mask_types):
                gen = torch.Generator(device=spikes.device).manual_seed(
                    session_idx * len(mask_types) + mode_idx
                )
                spikes_zeroed, is_mask, _ = self.masker(
                    ones, regions, mask_mode=mode, generator=gen
                )
                specs[(session_idx, mode)] = {
                    "is_mask": is_mask[0],  # (T, N)
                    "input_zero_mask": spikes_zeroed[0] == 0,  # (T, N)
                }
        return specs

    def _build_mode_metrics(self, mode, is_mask):
        """Create BPS and R2 metrics sized to the masked output dimension.

        causal        -> num_outputs = N        (all neurons, subset of timesteps)
        neuron/region -> num_outputs = N_masked (subset of neurons, all timesteps)

        Single GPU, which is all MTM has been run and tested on. The val loader is
        rank-sharded, so ranks would meet different sets of (mode, session) pairs, and
        torchmetrics' compute() syncs collectively: a pair only one rank holds leaves it
        waiting on a partner that never arrives. Pre-registering the metrics does not
        settle that on its own, because _precompute_val_masks walks the same sharded
        loader and needs a batch per session to size them. Going multi-GPU would need the
        session list taken from the dataset, num_outputs derived without a batch, and
        updated_keys reduced across ranks.
        """
        num_outputs = is_mask.shape[1] if mode == "causal" else int(is_mask[0, :].sum().item())
        return {
            "bps": BPS(num_outputs=num_outputs).to(self.device),
            "poisson_d2": PoissonD2Score(num_outputs=num_outputs).to(self.device),
        }

    def _update_mode_metrics(self, diag, mode, is_mask, pred_rates, target_spikes):
        """Update BPS/R2 for masked positions only, using mode-appropriate slicing.

        causal:        slice masked timesteps -> (B * masked_T, N)
        neuron/region: slice masked neurons   -> (B * T, N_masked)
        """
        pred_rates = pred_rates.float()
        if not torch.isfinite(pred_rates).all():
            return False
        if mode == "causal":
            mask_t = is_mask[:, 0]  # (T,): same for all neurons
            if not mask_t.any():
                return False
            flat_pred = pred_rates[:, mask_t, :].reshape(-1, pred_rates.shape[-1])
            flat_target = target_spikes[:, mask_t, :].reshape(-1, target_spikes.shape[-1]).long()
        else:
            mask_n = is_mask[0, :]  # (N,): same for all timesteps
            n_masked = int(mask_n.sum().item())
            if n_masked == 0:
                return False
            flat_pred = pred_rates[:, :, mask_n].reshape(-1, n_masked)
            flat_target = target_spikes[:, :, mask_n].reshape(-1, n_masked).long()
        for metric in diag.values():
            metric.update(flat_pred, flat_target)
        return True

    def _aggregate_mode_metrics(self, mode_session_diag, mask_types, updated_keys):
        """Average BPS/R2 over sessions for each mask mode, plus a grand average
        across all modes logged as {metric}/avg."""
        out = {}
        for metric_name in ("bps", "poisson_d2"):
            mode_avgs = []
            for mode in mask_types:
                vals = [
                    diag[metric_name].compute().item()
                    for key, diag in mode_session_diag.items()
                    if key[0] == mode and key in updated_keys
                ]
                finite = [v for v in vals if np.isfinite(v)]
                mode_avg = float(np.mean(finite)) if finite else float("nan")
                out[f"{metric_name}_{mode}/avg"] = mode_avg
                if np.isfinite(mode_avg):
                    mode_avgs.append(mode_avg)
            out[f"{metric_name}/avg"] = float(np.mean(mode_avgs)) if mode_avgs else float("nan")
        return out
