from __future__ import annotations

import multiprocessing as mp
from typing import Literal

import hydra
import numpy as np
import torch
from torch.nn import CrossEntropyLoss, MSELoss, PoissonNLLLoss
from torch.utils.data import DataLoader
from torch_brain.samplers import RandomFixedWindowSampler, TrialSampler
from torch_brain.transforms import Compose

from core.batching.collate import supervised_collate
from core.nn.loss import WeightedMSELoss
from core.samplers import DistributedSamplerWrapper
from core.trainer import BaseTrainer
from core.utils.exceptions import TrainingConstraintsError
from core.utils.util import get_num_params, get_num_trainable_params, human_readable, move_to_device
from ibl_bwb_eval.metrics import aggregate_metrics
from ibl_bwb_eval.tasks import DataType, TargetResolution, get_ts1_readout_spec
from pretrain.datasets import IBLBrainWideBenchSingleTaskBehavior

from .possm import POSSM


class POSSMSingleTaskPretrain(BaseTrainer):
    """Supervised pretraining of POSSM on one behavior."""

    # ------------------------------------------------------------
    # Setup trainer
    # ------------------------------------------------------------

    def setup(self, ckpt: dict | None):
        self.logger.info(f"Trainer: {self.__class__}")
        self.reset_best_tracking(minimize=False)

        # tasks
        self.setup_tasks()

        # data loaders
        self.setup_train_loader()
        self.setup_val_loader()

        # model
        self.setup_model(ckpt)
        self.link_model(self.model)

        # move model to device after linking, in case new parameters were created
        self.model.to(self.device)
        self.model = self.make_ddp(self.model)

        # optimizer and scheduler
        self.setup_optimizers(ckpt)

    def setup_tasks(self):
        self.task = self.cfg.task
        self.readout_spec = get_ts1_readout_spec(self.task)

        self.logger.info(f"Task: {self.task}")

        if (
            self.readout_spec.data_type == DataType.BINARY
            or self.readout_spec.data_type == DataType.MULTINOMIAL
        ):
            self.loss_fn = CrossEntropyLoss()
        elif self.readout_spec.data_type == DataType.CONTINUOUS:
            self.loss_fn = (
                WeightedMSELoss() if self.readout_spec.mask_key is not None else MSELoss()
            )
        elif self.readout_spec.data_type == DataType.EVENT_RATE:
            self.loss_fn = PoissonNLLLoss(log_input=True, full=True, eps=1e-9, reduction="mean")
        else:
            raise ValueError(f"Unsupported data type: {self.readout_spec.data_type}")

    def setup_train_loader(self):
        self.train_dataset = IBLBrainWideBenchSingleTaskBehavior(
            root=self.cfg.data_root,
            split="train",
            recording_ids=self.cfg.recording_ids,
            task=self.task,
        )
        self.train_sampler = DistributedSamplerWrapper(
            RandomFixedWindowSampler(
                sampling_intervals=self.train_dataset.get_trial_intervals(within_task_window=False),
                window_length=self.train_dataset.context_window,
                generator=torch.Generator().manual_seed(self.cfg.seed),
                drop_short=True,
            )
        )
        if len(self.train_sampler) < self.cfg.batch_size // self.world_size:
            raise TrainingConstraintsError(
                f"Batch size {self.cfg.batch_size // self.world_size} is larger than dataset size {len(self.train_sampler)}. "
                "No batches would be produced with drop_last=True. Please make sure the selected recording_ids "
                "contain enough samples with the targeted task(s)."
            )
        has_workers = self.cfg.num_workers > 0
        self.train_loader = DataLoader(
            self.train_dataset,
            sampler=self.train_sampler,
            collate_fn=supervised_collate,
            batch_size=self.cfg.batch_size // self.world_size,
            num_workers=self.cfg.num_workers,
            drop_last=True,
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
        self.val_dataset = IBLBrainWideBenchSingleTaskBehavior(
            root=self.cfg.data_root,
            split="val",
            recording_ids=self.cfg.recording_ids,
            task=self.task,
        )
        self.val_sampler = DistributedSamplerWrapper(
            TrialSampler(
                sampling_intervals=self.val_dataset.get_trial_intervals(within_task_window=True),
                shuffle=False,
            )
        )
        has_workers = self.cfg.num_workers > 0
        self.val_loader = DataLoader(
            self.val_dataset,
            sampler=self.val_sampler,
            collate_fn=supervised_collate,
            batch_size=self.cfg.batch_size // self.world_size,
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

    def setup_model(self, ckpt: dict | None):
        self.model = hydra.utils.instantiate(self.cfg.model)
        if not isinstance(self.model, POSSM):
            raise TypeError(
                f"{type(self).__name__} requires a POSSM model, got {type(self.model).__name__}"
            )
        self.add_checkpoint_items(model=self.model)

        self.logger.info(f"Precision: {self.precision}")
        self.logger.info(f"Model: {self.model.__class__}")
        self.model.train()

    def link_model(self, model: POSSM):
        # attach model transforms to end of dataset transform pipelines
        for split in ["train", "val"]:
            model_transforms = hydra.utils.instantiate(self.cfg.get(f"{split}_transforms", []))
            self.logger.info(f"Model transforms ({split}): {model_transforms}")
            dataset = getattr(self, f"{split}_loader").dataset

            if dataset.transform is None:
                dataset.transform = Compose([*model_transforms, model.input_fn])
            else:
                dataset.transform = Compose([dataset.transform, *model_transforms, model.input_fn])

        # link datasets to model
        model.link_datasets(
            self.train_loader.dataset,
            self.val_loader.dataset,
            test_dataset=None,
        )
        model.configure_readout(self.readout_spec)

        num_params, skipped_params = get_num_params(self.model)
        num_trainable_params = get_num_trainable_params(self.model, return_skipped=False)
        num_embedding_params = get_num_params(
            self.model.unit_emb, return_skipped=False
        ) + get_num_params(self.model.session_emb, return_skipped=False)
        self.logger.info(f"Number of parameters: {human_readable(num_params)} ({num_params:,})")
        self.logger.info(
            f"Number of trainable parameters: {human_readable(num_trainable_params)} ({num_trainable_params:,})"
        )
        self.logger.info(
            f"Number of embedding parameters: {human_readable(num_embedding_params)} ({num_embedding_params:,})"
        )
        if skipped_params:
            self.logger.info(f"Skipped lazy parameters: {skipped_params}")

    def setup_optimizers(self, ckpt: dict | None):
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

    # ------------------------------------------------------------
    # Training / evaluation
    # ------------------------------------------------------------

    def loss(self, pred, X, y):
        target = y["values"]
        if self.readout_spec.target_resolution == TargetResolution.TIMESTEP:
            # (B, T, D) -> (B, T*D)
            pred = pred.view(pred.shape[0], -1)
            target = target.view(target.shape[0], -1)
        else:
            # (B, 1, D) -> (B, D)
            pred = pred.squeeze(1)
            # (B, 1) -> (B,) for classification loss
            target = target.squeeze(-1)

        if "mask" in y:
            assert self.readout_spec.target_resolution == TargetResolution.TIMESTEP, (
                "Target mask is only supported for timestep-level tasks"
            )
            # (B, T) -> (B, T, 1) -> (B, T, D) -> (B, T*D)
            mask = y["mask"]
            mask = mask.unsqueeze(-1).expand(-1, -1, self.readout_spec.dim)
            mask = mask.reshape(mask.shape[0], -1)
            # compute loss with mask
            return self.loss_fn(pred, target, mask)
        else:
            return self.loss_fn(pred, target)

    def train_epoch(self):
        self.model.train()
        loader = self.train_loader
        if self.cfg.log_train_step:
            loader = self.logger.get_pbar(loader, prefix="train")

        epoch_losses = []
        for X, y in loader:
            X = move_to_device(X, device=self.device)
            y = move_to_device(y, device=self.device)
            self.optimizer.zero_grad()
            with torch.autocast(device_type="cuda", dtype=self.precision.dtype):
                # POSSM input_fn prepares all output queries inside model_inputs.
                pred = self.model(**X["model_inputs"])
                loss = self.loss(pred, X, y)
            loss.backward()
            self.clip_and_log_grad_norm()
            self.optimizer.step()
            self.scheduler.step()
            self.train_step += 1

            loss_val = loss.item()
            epoch_losses.append(loss_val)

            if self.cfg.log_train_step:
                # log loss every step
                self.logger.log("train/loss", loss_val, pbar=True)
                self.logger.log("train/lr", self.scheduler.get_last_lr()[0], pbar=False)
                self.push_logs()

        if not self.cfg.log_train_step:
            # log average loss over train epoch
            self.logger.log("train/loss", sum(epoch_losses) / len(epoch_losses))

    @torch.inference_mode()
    def val_epoch(self):
        self.model.eval()
        val_metrics = self._eval_epoch(self.val_loader, split="val")
        selected = val_metrics[self.readout_spec.primary_metric]

        self.save_if_best(selected, metrics=val_metrics)

        log_metrics = {f"val/{k}": v for k, v in val_metrics.items()} | {
            "val/avg": selected,
            "val/epoch": self.epoch,
        }
        self.logger.log_dict(log_metrics)
        self.push_logs()
        self.return_value = log_metrics

    def _eval_epoch(self, loader: DataLoader, split: Literal["val"] = "val"):
        # general evaluation epoch (e.g. validation)
        with torch.no_grad():
            metrics = {
                name: metric().to(self.device) for name, metric in self.readout_spec.metrics.items()
            }

            for X, y in self.logger.get_pbar(loader, prefix=split):
                X = move_to_device(X, device=self.device)
                y = move_to_device(y, device=self.device)
                with torch.autocast(device_type="cuda", dtype=self.precision.dtype):
                    # POSSM input_fn prepares all output queries (timestamps,
                    # decoder_index, bin_index, session_index) inside model_inputs.
                    pred = self.model(**X["model_inputs"])
                    target = y["values"]

                    if self.readout_spec.target_resolution == TargetResolution.TIMESTEP:
                        # (B, T, D) -> (B*T, D)
                        pred = pred.view(-1, self.readout_spec.dim)
                        target = target.view(-1, self.readout_spec.dim)
                    else:
                        # (B, 1, D) -> (B, D)
                        pred = pred.squeeze(1)

                for metric in metrics.values():
                    if "mask" in y:
                        mask = y["mask"].flatten()  # (B*T,)
                        assert mask.dtype == torch.bool, "Mask must be a boolean tensor"
                        if mask.sum() == 0:
                            continue
                        metric.update(pred[mask], target[mask])  # (B*T,)
                    elif (
                        self.readout_spec.data_type == DataType.BINARY
                        or self.readout_spec.data_type == DataType.MULTINOMIAL
                    ):
                        # for K-way classification, ensure targets are of shape (B,)
                        # (label indices) by removing the last singleton dimension
                        metric.update(pred, target.squeeze(-1))
                    else:
                        metric.update(pred, target)

            eval_metrics = aggregate_metrics(metrics)

        return eval_metrics
