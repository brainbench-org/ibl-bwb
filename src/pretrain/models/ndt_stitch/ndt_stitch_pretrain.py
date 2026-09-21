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

from .masker import NDTStitchMasker
from .ndt_stitch import NDTStitch


class NDTStitchPretrain(BaseTrainer):
    """Masked spike modelling for NDT Stitch, which fits one stitcher per session."""

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
        if not isinstance(self.model, NDTStitch):
            raise TypeError(
                f"{type(self).__name__} requires a NDTStitch model, got {type(self.model).__name__}"
            )
        self.masker = hydra.utils.instantiate(self.cfg.masker)
        if not isinstance(self.masker, NDTStitchMasker):
            raise TypeError(
                f"{type(self).__name__} requires a NDTStitchMasker masker, got {type(self.masker).__name__}"
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

                masked_spikes, mask = self.masker(spikes)
                X["model_inputs"]["spikes"] = masked_spikes

                pred_rates = self.predict(X)

                loss = self.loss(pred_rates, target_spikes, mask)

            loss.backward()
            self.clip_and_log_grad_norm()
            self.optimizer.step()
            self.scheduler.step()
            self.train_step += 1

            if self.cfg.log_train_step:
                self.logger.log("train/loss", loss.item(), pbar=True)
                self.logger.log("train/lr", self.scheduler.get_last_lr()[0], pbar=False)
                self.push_logs()

    def predict(self, X):
        return self.model(**X["model_inputs"])

    def loss(self, pred_rates, target_spikes, mask):
        loss = self.loss_fn(pred_rates, target_spikes)
        if mask.sum() == 0:
            return loss.sum() * 0.0
        return loss[mask].mean()

    def _build_session_metrics(self):
        """Build per-session BPS / Cohen pseudo-R2 metric dicts.
        Batches are single-session (SessionBatchSampler), so updates are dispatched
        by the unique session token in each batch.

        Per-session ``num_outputs`` is passed eagerly so the metric states are
        registered up-front on every rank, required for DDP collectives, since
        otherwise lazy init means a rank that never sees a given session ends up
        without those states and the all_reduce in ``compute()`` mis-syncs.
        """
        num_units = {
            **self.train_dataset.get_num_unit_per_recording(),
            **self.val_dataset.get_num_unit_per_recording(),
        }
        session_vocab = {
            k: v for k, v in self.model.session_emb.vocab.items() if k != "NA" and k in num_units
        }
        diagnostic_metrics = {
            "poisson_d2": {
                idx: PoissonD2Score(num_outputs=num_units[name]).to(self.device)
                for name, idx in session_vocab.items()
            },
            "bps": {
                idx: BPS(num_outputs=num_units[name]).to(self.device)
                for name, idx in session_vocab.items()
            },
        }
        return session_vocab, diagnostic_metrics

    def _update_session_metrics(self, diagnostic_metrics, X, pred_rates, target_spikes, mask=None):
        """Update per-session metrics for one batch, False if non-finite rates dropped it.

        If mask (B, T, N) is provided, only masked timesteps are evaluated.
        """
        pred_rates = pred_rates.float()
        session_idx = int(X["model_inputs"]["session_tokens"].unique().item())
        if mask is not None:
            # mask is (B, T, N) with all-neurons-equal per timestep; collapse to (B, T)
            mask_t = mask[:, :, 0]
            flat_pred = pred_rates[mask_t]  # (num_masked_timesteps, N)
            flat_target = target_spikes[mask_t].long()
        else:
            B, T, N = pred_rates.shape
            flat_pred = pred_rates.reshape(B * T, N)
            flat_target = target_spikes.reshape(B * T, N).long()

        # only the scored positions reach the metric
        if not torch.isfinite(flat_pred).all():
            return False

        for metric_dict in diagnostic_metrics.values():
            metric_dict[session_idx].update(flat_pred, flat_target)
        return True

    def _aggregate_session_metrics(self, session_vocab, diagnostic_metrics):
        """Compute per-session values + cross-session mean, return a flat dict.

        The mean covers the sessions that scored finite; ``{metric}/n_sessions``
        says how many, since the mean alone reads the same over 400 or over 4.
        """
        per_session = {
            k: {name: d[idx].compute().item() for name, idx in session_vocab.items()}
            for k, d in diagnostic_metrics.items()
        }
        out = {}
        for k, sess_vals in per_session.items():
            out.update({f"{k}/{name}": v for name, v in sess_vals.items()})
            finite = [v for v in sess_vals.values() if np.isfinite(v)]
            out[f"{k}/avg"] = float(np.mean(finite)) if finite else float("nan")
            out[f"{k}/n_sessions"] = len(finite)
            if len(finite) < len(sess_vals):
                self.logger.warn(f"val/{k}/avg covers {len(finite)} of {len(sess_vals)} sessions")
        return out

    @torch.inference_mode()
    def val_epoch(self):
        self.model.eval()
        session_vocab, diagnostic_metrics = self._build_session_metrics()

        loss_sum, loss_count = 0.0, 0
        n_batches, n_dropped = 0, 0
        for X in self.val_loader:
            X = move_to_device(X, device=self.device)
            with torch.autocast(device_type="cuda", dtype=self.precision.dtype):
                spikes = X["model_inputs"]["spikes"]
                target_spikes = spikes.clone()

                masked_spikes, mask = self.masker(spikes)
                X["model_inputs"]["spikes"] = masked_spikes

                pred_rates = self.predict(X)

                loss = self.loss(pred_rates, target_spikes, mask)
                mask_count = int(mask.sum().item())
                loss_sum += float(loss.item()) * mask_count
                loss_count += mask_count

                n_batches += 1
                if not self._update_session_metrics(
                    diagnostic_metrics, X, pred_rates, target_spikes, mask
                ):
                    n_dropped += 1

        if n_dropped:
            self.logger.warn(
                f"val: {n_dropped} of {n_batches} batches dropped from the diagnostic "
                "metrics for non-finite rates"
            )

        val_loss = self.reduce_mean(loss_sum, loss_count)
        metrics = {
            "avg": val_loss,
            "loss": val_loss,
            "epoch": self.epoch,
        }
        metrics.update(self._aggregate_session_metrics(session_vocab, diagnostic_metrics))

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
