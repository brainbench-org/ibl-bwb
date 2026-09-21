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

from core.samplers import DistributedSamplerWrapper
from core.trainer import BaseTrainer
from core.utils.util import log_param_breakdown, move_to_device
from pretrain.datasets import IBLBrainWideBenchMaskModelingSpikes

from .masker import NDT2Masker
from .ndt2 import NDT2


class NDT2Pretrain(BaseTrainer):
    """Masked spike modelling for NDT2, which masks context tokens and accumulates gradients."""

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

        self.train_sampler = DistributedSamplerWrapper(
            RandomFixedWindowSampler(
                sampling_intervals=self.train_dataset.get_trial_intervals(),
                window_length=self.train_dataset.context_window,
                generator=torch.Generator().manual_seed(self.cfg.seed),
                drop_short=True,
            )
        )

        has_workers = self.cfg.num_workers > 0
        # cfg.batch_size is the global batch, so each rank draws its own share.
        self.train_loader = DataLoader(
            self.train_dataset,
            sampler=self.train_sampler,
            batch_size=self.cfg.batch_size // self.world_size,
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
            f"Training on {len(self.train_loader)} batches, total {len(self.train_sampler)} samples"
        )

    def setup_val_loader(self):
        self.val_dataset = IBLBrainWideBenchMaskModelingSpikes(
            root=self.cfg.data_root,
            recording_ids=self.cfg.recording_ids,
            split="val",
        )

        self.val_sampler = DistributedSamplerWrapper(
            SequentialFixedWindowSampler(
                sampling_intervals=self.val_dataset.get_trial_intervals(),
                window_length=self.val_dataset.context_window,
                drop_short=True,
            ),
        )

        has_workers = self.cfg.num_workers > 0
        self.val_loader = DataLoader(
            self.val_dataset,
            sampler=self.val_sampler,
            batch_size=self.cfg.batch_size // self.world_size,
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
            f"Validating on {len(self.val_loader)} batches, total {len(self.val_sampler)} samples"
        )

    def setup_model(self, ckpt: dict | None = None):
        self.model = hydra.utils.instantiate(self.cfg.model)
        if not isinstance(self.model, NDT2):
            raise TypeError(
                f"{type(self).__name__} requires a NDT2 model, got {type(self.model).__name__}"
            )
        self.masker = hydra.utils.instantiate(self.cfg.masker)
        if not isinstance(self.masker, NDT2Masker):
            raise TypeError(
                f"{type(self).__name__} requires a NDT2Masker masker, got {type(self.masker).__name__}"
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
        # ckpt is set by ckpt.load_from, whether or not the run resumes epoch and step
        if ckpt is not None:
            model.load_ckpt(ckpt)

        log_param_breakdown(self.logger, self.model, ("session_emb", "subject_emb", "task_emb"))

    def setup_optimizers(self, ckpt: dict | None = None):
        if self.model is None:
            raise ValueError("Trying to configure optimizers before the model is setup")
        if self.train_loader is None:
            raise ValueError("Trying to configure optimizers before the train data loader is setup")

        grouped_parameters = self.get_param_groups()

        max_lr = self.cfg.base_lr * np.sqrt(self.cfg.batch_size * self.cfg.grad_accum_steps)
        self.logger.info(f"Max LR: {max_lr}")
        self.optimizer = torch.optim.AdamW(
            grouped_parameters,
            lr=max_lr,
        )
        self.scheduler = torch.optim.lr_scheduler.OneCycleLR(
            self.optimizer,
            max_lr=max_lr,
            epochs=self.cfg.num_epochs,
            steps_per_epoch=int(np.ceil(len(self.train_loader) / self.cfg.grad_accum_steps)),
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
        loader_len = len(loader)
        if self.cfg.log_train_step:
            loader = self.logger.get_pbar(loader, prefix="train")

        for i, X in enumerate(loader):
            X = move_to_device(X, device=self.device)
            with torch.autocast(device_type="cuda", dtype=self.precision.dtype):
                patches = X["model_inputs"]["in_patches"]
                ssl_mask = self.masker(patches)
                pred_rates = self.predict(X, ssl_mask)

                # `ssl_mask` selects which patches were hidden and are therefore queried;
                # `loss_mask` says which of those queried bins are scored.
                B, L_q, P = pred_rates.shape
                target = X["ssl"]["target"]
                target_spikes = target[ssl_mask].reshape(B, L_q, P).float()

                in_not_pad = X["model_inputs"]["in_not_pad"]
                query_not_pad = in_not_pad[ssl_mask].reshape(B, L_q).unsqueeze(-1)
                query_is_valid = X["ssl"]["is_valid"][ssl_mask].reshape(B, L_q, P)
                loss_mask = query_not_pad & query_is_valid

                unscaled_loss = self.loss(pred_rates, target_spikes, loss_mask)

            should_step = (
                (i + 1) % self.cfg.grad_accum_steps == 0  # is accum step
            ) or (
                (i + 1) == loader_len  # is last batch
            )

            loss = unscaled_loss / self.cfg.grad_accum_steps
            # DDP all-reduces on every backward, so the accumulation steps in between
            # pay for gradients no one reads until the step that syncs them.
            if self.is_distributed and not should_step:
                with self.model.no_sync():
                    loss.backward()
            else:
                loss.backward()

            if should_step:
                self.clip_and_log_grad_norm()
                self.optimizer.step()
                self.scheduler.step()
                self.optimizer.zero_grad()
                self.train_step += 1

            if self.cfg.log_train_step:
                # log loss every step
                self.logger.log("train/loss", unscaled_loss.item(), pbar=True)
                self.logger.log("train/lr", self.scheduler.get_last_lr()[0], pbar=False)
                self.push_logs()

    def predict(self, X, ssl_mask):
        rates = self.model(**X["model_inputs"], ssl_mask=ssl_mask)
        return rates

    def loss(self, pred_rates, target_spikes, loss_mask):
        loss = self.loss_fn(pred_rates, target_spikes)
        return loss[loss_mask].mean()

    @torch.inference_mode()
    def val_epoch(self):
        self.model.eval()

        loss_sum, loss_count = 0.0, 0
        for X in self.logger.get_pbar(self.val_loader, prefix="val"):
            X = move_to_device(X, device=self.device)
            with torch.autocast(device_type="cuda", dtype=self.precision.dtype):
                patches = X["model_inputs"]["in_patches"]
                ssl_mask = self.masker(patches)
                pred_rates = self.predict(X, ssl_mask)

                B, L_q, P = pred_rates.shape
                target = X["ssl"]["target"]
                target_spikes = target[ssl_mask].reshape(B, L_q, P).float()

                in_not_pad = X["model_inputs"]["in_not_pad"]
                query_not_pad = in_not_pad[ssl_mask].reshape(B, L_q).unsqueeze(-1)

                query_is_valid = X["ssl"]["is_valid"][ssl_mask].reshape(B, L_q, P)

                loss_mask = query_not_pad & query_is_valid

                loss = self.loss(pred_rates, target_spikes, loss_mask)

                n_scored = int(loss_mask.sum().item())
                loss_sum += float(loss.item()) * n_scored
                loss_count += n_scored

        val_loss = self.reduce_mean(loss_sum, loss_count)
        metrics = {
            "avg": val_loss,
            "loss": val_loss,
            "epoch": self.epoch,
        }
        self.save_if_best(val_loss, metrics=metrics)

        # store and log metrics
        val_metrics = {f"val/{k}": v for k, v in metrics.items()}
        self.logger.log_dict(val_metrics)
        self.push_logs()
        self.logger.update_epoch_postfix(self.get_best_pbar_metrics())
        self.return_value = val_metrics

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

    def _setup_ssl_task(self):
        self.logger.info("Task: Mask Modeling SSL (MAE style)")
        self.loss_fn = PoissonNLLLoss(log_input=True, full=True, eps=1e-9, reduction="none")
