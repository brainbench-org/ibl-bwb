from __future__ import annotations

import math
import multiprocessing as mp

import hydra
import numpy as np
import torch
import torch.nn as nn
from omegaconf import ListConfig, OmegaConf
from rich.console import Console
from rich.table import Table
from torch.nn import CrossEntropyLoss, MSELoss, PoissonNLLLoss
from torch.utils.data import DataLoader
from torch_brain.batching import collate
from torch_brain.samplers import RandomFixedWindowSampler, TrialSampler
from torch_brain.transforms.container import Compose
from torchmetrics import Metric

from core.nn.loss import WeightedMSELoss
from core.samplers import DistributedSamplerWrapper
from core.trainer import BaseTrainer
from core.utils.exceptions import TrainingConstraintsError
from core.utils.util import get_num_params, get_num_trainable_params, human_readable, move_to_device
from ibl_bwb_eval.tasks import (
    DataType,
    TS1ReadoutSpec,
    get_ts1_readout_spec,
    get_ts1_supported_tasks,
)
from pretrain.datasets.multi_task_behavior import IBLBrainWideBenchMultiTaskBehavior

from .possm import POSSM


class POSSMMultitaskPretrain(BaseTrainer):
    """Multi-task pretrainer for POSSM.

    Trains a single POSSM model on multiple decoding tasks simultaneously,
    using its :class:`MultitaskReadout`. Per-task targets are gathered by
    the model's ``input_fn`` (no per-sample dataset target extraction).
    Supports per-task eval metrics, configurable best-model selection (loss
    or metric), held-out test, and full ckpt resume.
    """

    # ------------------------------------------------------------
    # Setup
    # ------------------------------------------------------------

    def setup(self, ckpt: dict | None):
        self.logger.info(f"Trainer: {self.__class__}")
        assert self.cfg.val.selection_key in ("avg_metric", "avg_loss", "weighted_avg_loss"), (
            f"unsupported val.selection_key={self.cfg.val.selection_key}"
        )
        self._setup_tasks()

        self.setup_train_loader()
        self.setup_val_loader()

        self.setup_model()
        self.link_model(self.model)
        self._restore_model_from_ckpt(ckpt)

        self.model.to(self.device)
        self.model = self.make_ddp(self.model)

        self.setup_optimizers(ckpt)
        self.add_checkpoint_items(
            model=self.model,
            optimizer=self.optimizer,
            scheduler=self.scheduler,
        )

    def _setup_tasks(self):
        tasks = self.cfg.get("tasks", None)
        if isinstance(tasks, ListConfig):
            tasks = list(tasks)
        if tasks is None:
            tasks = get_ts1_supported_tasks()
        elif not isinstance(tasks, list):
            tasks = [tasks]

        self.tasks = list(tasks)
        self.readout_specs: dict[str, TS1ReadoutSpec] = {
            task: get_ts1_readout_spec(task) for task in self.tasks
        }
        self.loss_fns: dict[str, nn.Module] = {
            task: self._build_loss_fn(task) for task in self.tasks
        }

        # Per-task loss weights. Tasks not listed in cfg.task_weights default
        # to 1.0. Used as multiplicative scalars in compute_multitask_loss.
        cfg_weights = self.cfg.get("task_weights", None)
        if cfg_weights is None:
            cfg_weights = {}
        elif OmegaConf.is_config(cfg_weights):
            cfg_weights = OmegaConf.to_container(cfg_weights, resolve=True)
        self.task_weights: dict[str, float] = {
            task: float(cfg_weights.get(task, 1.0)) for task in self.tasks
        }
        self.logger.info(f"Multitask pretraining tasks: {self.tasks}")
        self.logger.info(
            "Task weights: " + ", ".join(f"{t}={w:g}" for t, w in self.task_weights.items())
        )

    def _build_loss_fn(self, task_name: str):
        spec = self.readout_specs[task_name]
        if spec.data_type in {DataType.BINARY, DataType.MULTINOMIAL}:
            return CrossEntropyLoss()
        if spec.data_type == DataType.CONTINUOUS:
            return WeightedMSELoss() if spec.mask_key is not None else MSELoss()
        if spec.data_type == DataType.EVENT_RATE:
            return PoissonNLLLoss(log_input=True, full=True, eps=1e-9, reduction="mean")
        raise ValueError(f"Unsupported data type: {spec.data_type}")

    # ------------------------------------------------------------
    # Data
    # ------------------------------------------------------------

    def _make_loader(self, split: str, sampler):
        has_workers = self.cfg.num_workers > 0
        return DataLoader(
            getattr(self, f"{split}_dataset"),
            sampler=sampler,
            collate_fn=collate,
            batch_size=self.cfg.batch_size // self.world_size,
            num_workers=self.cfg.num_workers,
            drop_last=(split == "train"),
            pin_memory=self.cfg.pin_memory,
            prefetch_factor=self.cfg.prefetch_factor if has_workers else None,
            persistent_workers=self.cfg.persistent_workers if has_workers else False,
            multiprocessing_context=mp.get_context("fork") if has_workers else None,
        )

    def setup_train_loader(self):
        self.train_dataset = IBLBrainWideBenchMultiTaskBehavior(
            root=self.cfg.data_root,
            split="train",
            recording_ids=self.cfg.recording_ids,
            tasks=self.tasks,
        )
        self.train_sampler = DistributedSamplerWrapper(
            RandomFixedWindowSampler(
                sampling_intervals=self.train_dataset.get_trial_intervals(),
                window_length=self.train_dataset.context_window,
                generator=torch.Generator().manual_seed(self.cfg.seed),
                drop_short=True,
            )
        )
        if len(self.train_sampler) < self.cfg.batch_size // self.world_size:
            raise TrainingConstraintsError(
                f"Batch size {self.cfg.batch_size // self.world_size} is larger than "
                f"dataset size {len(self.train_sampler)}."
            )
        self.train_loader = self._make_loader("train", self.train_sampler)
        self.logger.info(
            f"Train dataset: {len(self.train_dataset.get_session_ids())} sessions, "
            f"{len(self.train_dataset.get_unit_ids())} units"
        )
        self.logger.info(
            f"Training on {len(self.train_loader)} batches, total {len(self.train_sampler)} samples"
        )

    def setup_val_loader(self):
        self.val_dataset = IBLBrainWideBenchMultiTaskBehavior(
            root=self.cfg.data_root,
            split="val",
            recording_ids=self.cfg.recording_ids,
            tasks=self.tasks,
        )
        self.val_sampler = DistributedSamplerWrapper(
            TrialSampler(
                sampling_intervals=self.val_dataset.get_task_align_intervals(),
                shuffle=False,
            )
        )
        self.val_loader = self._make_loader("val", self.val_sampler)
        self.logger.info(
            f"Val dataset: {len(self.val_dataset.get_session_ids())} sessions, "
            f"{len(self.val_dataset.get_unit_ids())} units"
        )
        self.logger.info(
            f"Validating on {len(self.val_loader)} batches, total {len(self.val_sampler)} samples"
        )

    def setup_test_loader(self):
        self.test_dataset = IBLBrainWideBenchMultiTaskBehavior(
            root=self.cfg.data_root,
            split="test",
            recording_ids=self.cfg.recording_ids,
            tasks=self.tasks,
        )
        self.test_sampler = DistributedSamplerWrapper(
            TrialSampler(
                sampling_intervals=self.test_dataset.get_task_align_intervals(),
                shuffle=False,
            )
        )
        self.test_loader = self._make_loader("test", self.test_sampler)
        self.logger.info(
            f"Test dataset: {len(self.test_dataset.get_session_ids())} sessions, "
            f"{len(self.test_dataset.get_unit_ids())} units"
        )
        self.logger.info(
            f"Testing on {len(self.test_loader)} batches, total {len(self.test_sampler)} samples"
        )

    # ------------------------------------------------------------
    # Model + optimizer
    # ------------------------------------------------------------

    def setup_model(self):
        self.model = hydra.utils.instantiate(self.cfg.model)
        if not isinstance(self.model, POSSM):
            raise TypeError(
                f"{type(self).__name__} requires a POSSM model, got {type(self.model).__name__}"
            )
        self.logger.info(f"Precision: {self.precision}")
        self.logger.info(f"Model: {self.model.__class__}")
        self.model.train()

    def link_model(self, model: POSSM):
        for split in ["train", "val"]:
            model_transforms = hydra.utils.instantiate(self.cfg.get(f"{split}_transforms", []))
            self.logger.info(f"Model transforms ({split}): {model_transforms}")
            dataset = getattr(self, f"{split}_loader").dataset

            if dataset.transform is None:
                dataset.transform = Compose([*model_transforms, model.input_fn])
            else:
                dataset.transform = Compose([dataset.transform, *model_transforms, model.input_fn])

        model.link_datasets(
            self.train_loader.dataset,
            self.val_loader.dataset,
        )
        model.configure_multitask_readout(self.readout_specs)

        num_params, skipped_params = get_num_params(self.model)
        num_trainable_params = get_num_trainable_params(self.model, return_skipped=False)
        num_embedding_params = get_num_params(
            self.model.unit_emb, return_skipped=False
        ) + get_num_params(self.model.session_emb, return_skipped=False)
        self.logger.info(f"Number of parameters: {human_readable(num_params)} ({num_params:,})")
        self.logger.info(
            f"Number of trainable parameters: "
            f"{human_readable(num_trainable_params)} ({num_trainable_params:,})"
        )
        self.logger.info(
            f"Number of embedding parameters: "
            f"{human_readable(num_embedding_params)} ({num_embedding_params:,})"
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
        return self.optimizer, self.scheduler

    # ------------------------------------------------------------
    # Checkpoint restore
    # ------------------------------------------------------------

    def _restore_model_from_ckpt(self, ckpt: dict | None):
        if ckpt is None or "model_state_dict" not in ckpt:
            return

        strict = bool(self.cfg.ckpt.resume) and not bool(self.cfg.ckpt.model_only)
        state_dict = ckpt["model_state_dict"]
        try:
            if strict:
                self.model.load_state_dict(state_dict, strict=True)
                self.logger.info("Loaded model checkpoint with strict=True")
            else:
                incompat = self.model.load_state_dict(state_dict, strict=False)
                if incompat.missing_keys:
                    self.logger.warning(
                        f"Missing model keys when loading checkpoint: {len(incompat.missing_keys)}"
                    )
                if incompat.unexpected_keys:
                    self.logger.warning(
                        f"Unexpected model keys when loading checkpoint: "
                        f"{len(incompat.unexpected_keys)}"
                    )
                self.logger.info("Loaded model checkpoint with strict=False")
        except RuntimeError as exc:
            self.logger.warning(
                f"Strict checkpoint load failed ({exc}). Retrying with strict=False."
            )
            incompat = self.model.load_state_dict(state_dict, strict=False)
            if incompat.missing_keys:
                self.logger.warning(
                    f"Missing model keys when loading checkpoint: {len(incompat.missing_keys)}"
                )
            if incompat.unexpected_keys:
                self.logger.warning(
                    f"Unexpected model keys when loading checkpoint: "
                    f"{len(incompat.unexpected_keys)}"
                )
            self.logger.info("Loaded model checkpoint with strict=False")

    # ------------------------------------------------------------
    # Loss + metrics
    # ------------------------------------------------------------

    @staticmethod
    def _unwrap_chained(x):
        """``chain(...)`` outputs become tuples ``(tensor, batch_index)`` after
        collation; we only need the tensor for loss / metric computation."""
        return x[0] if isinstance(x, tuple) else x

    def _task_loss(self, task_name: str, pred: torch.Tensor, batch: dict):
        target_values = batch.get("target_values", {})
        if task_name not in target_values:
            return None

        spec = self.readout_specs[task_name]
        target = self._unwrap_chained(target_values[task_name])

        if spec.data_type in {DataType.BINARY, DataType.MULTINOMIAL}:
            target = target.reshape(-1).long()
            return self.loss_fns[task_name](pred.reshape(-1, spec.dim), target)

        pred = pred.reshape(-1, spec.dim)
        target = target.reshape(-1, spec.dim).float()

        target_masks = batch.get("target_masks", {})
        if spec.data_type == DataType.CONTINUOUS and task_name in target_masks:
            mask = self._unwrap_chained(target_masks[task_name]).bool()
            if mask.ndim == 1:
                mask = mask.unsqueeze(-1).expand(-1, spec.dim)
            mask = mask.reshape(-1, spec.dim).float()
            if mask.sum() <= 0:
                return None
            return self.loss_fns[task_name](pred, target, mask)
        return self.loss_fns[task_name](pred, target)

    def compute_multitask_loss(self, pred_dict: dict[str, torch.Tensor], batch: dict):
        """Aggregate per-task losses into a single training scalar.

        Returns the unweighted ``task_losses`` for logging / metrics, plus a
        weighted sum ``sum_t (task_weights[t] * task_losses[t])`` as the
        backward signal. Weighted-sum (not weighted-mean) is intentional:
        scaling a task's weight up directly increases its gradient
        contribution, matching how POSSM's recommended task_weights are
        usually written down.
        """
        task_losses: dict[str, torch.Tensor] = {}
        for task_name, pred in pred_dict.items():
            loss = self._task_loss(task_name, pred, batch)
            if loss is None:
                continue
            task_losses[task_name] = loss
        if not task_losses:
            return None, {}
        total = torch.stack(
            [task_losses[t] * self.task_weights.get(t, 1.0) for t in task_losses]
        ).sum()
        return total, task_losses

    def _build_eval_metrics(self):
        eval_metrics: dict[str, dict[str, Metric]] = {}
        for task_name in self.tasks:
            eval_metrics[task_name] = {
                metric_name: metric_ctor().to(self.device)
                for metric_name, metric_ctor in self.readout_specs[task_name].metrics.items()
            }
        return eval_metrics

    def _prepare_metric_inputs(self, task_name: str, pred: torch.Tensor, batch: dict):
        target_values = batch.get("target_values", {})
        if task_name not in target_values:
            return None, None

        spec = self.readout_specs[task_name]
        target = self._unwrap_chained(target_values[task_name])

        if spec.data_type in {DataType.BINARY, DataType.MULTINOMIAL}:
            target = target.reshape(-1).long()
            if target.numel() == 0:
                return None, None
            return pred.reshape(-1, spec.dim), target

        pred = pred.reshape(-1, spec.dim)
        target = target.reshape(-1, spec.dim).float()

        target_masks = batch.get("target_masks", {})
        if spec.data_type == DataType.CONTINUOUS and task_name in target_masks:
            mask = self._unwrap_chained(target_masks[task_name]).bool()
            if mask.ndim == 1:
                mask = mask.unsqueeze(-1).expand(-1, spec.dim)
            mask = mask.reshape(-1, spec.dim)
            valid = mask.any(dim=-1)
            if valid.sum() <= 0:
                return None, None
            pred = pred[valid]
            target = target[valid]

        if target.numel() == 0:
            return None, None
        return pred, target

    @staticmethod
    def _to_scalar_metric(v):
        if torch.is_tensor(v):
            return v.mean().item() if v.ndim > 0 else v.item()
        return float(v)

    # ------------------------------------------------------------
    # Training / evaluation
    # ------------------------------------------------------------

    def predict(self, batch):
        return self.model(**batch["model_inputs"], return_dict=True)

    def train_epoch(self):
        self.model.train()
        loader = self.train_loader
        if self.cfg.log_train_step:
            loader = self.logger.get_pbar(loader, prefix="train")

        for batch in loader:
            batch = move_to_device(batch, device=self.device)
            self.optimizer.zero_grad()
            with torch.autocast(device_type="cuda", dtype=self.precision.dtype):
                pred_dict = self.predict(batch)
                total_loss, task_losses = self.compute_multitask_loss(pred_dict, batch)

            if total_loss is None:
                continue

            total_loss.backward()
            self.clip_and_log_grad_norm()
            self.optimizer.step()
            self.scheduler.step()
            self.train_step += 1

            if self.cfg.log_train_step:
                self.logger.log("train/loss", total_loss.item(), pbar=True)
                self.logger.log("train/lr", self.scheduler.get_last_lr()[0], pbar=False)
                for task_name, loss in task_losses.items():
                    self.logger.log(f"train/{task_name}_loss", loss.item())
                self.push_logs()

    @torch.inference_mode()
    def _eval_epoch(self, loader: DataLoader, split: str):
        loss_sums: dict[str, float] = dict.fromkeys(self.tasks, 0.0)
        loss_counts: dict[str, int] = dict.fromkeys(self.tasks, 0)
        weighted_total_sum = 0.0
        weighted_total_count = 0
        eval_metrics = self._build_eval_metrics()
        valid_metric_tasks = set()

        for batch in self.logger.get_pbar(loader, prefix=split):
            batch = move_to_device(batch, device=self.device)
            with torch.autocast(device_type="cuda", dtype=self.precision.dtype):
                pred_dict = self.predict(batch)
                total_loss, task_losses = self.compute_multitask_loss(pred_dict, batch)
            if total_loss is None:
                continue
            weighted_total_sum += total_loss.item()
            weighted_total_count += 1
            for task_name, loss in task_losses.items():
                loss_sums[task_name] += loss.item()
                loss_counts[task_name] += 1
            for task_name, pred in pred_dict.items():
                if task_name not in eval_metrics:
                    continue
                pred_eval, target_eval = self._prepare_metric_inputs(task_name, pred, batch)
                if pred_eval is None:
                    continue
                valid_metric_tasks.add(task_name)
                for metric in eval_metrics[task_name].values():
                    metric.update(pred_eval, target_eval)

        pooled = {t: self.reduce_mean(loss_sums[t], loss_counts[t]) for t in self.tasks}
        metrics: dict[str, float] = {
            f"{task}_loss": v for task, v in pooled.items() if not math.isnan(v)
        }
        for task_name in self.tasks:
            if task_name not in valid_metric_tasks:
                continue
            for metric_name, metric in eval_metrics[task_name].items():
                metrics[f"{task_name}/{metric_name}"] = self._to_scalar_metric(metric.compute())

        # avg_loss = unweighted mean over per-task average losses, so it stays
        # comparable across runs with different task_weights.
        per_task_means = [
            metrics[f"{task}_loss"] for task in self.tasks if f"{task}_loss" in metrics
        ]
        metrics["avg_loss"] = (
            float(sum(per_task_means) / len(per_task_means)) if per_task_means else float("inf")
        )
        # weighted_avg_loss = mean over batches of the weighted training scalar.
        weighted = self.reduce_mean(weighted_total_sum, weighted_total_count)
        metrics["weighted_avg_loss"] = float("inf") if math.isnan(weighted) else weighted
        primary_values = []
        for task_name in self.tasks:
            primary_name = self.readout_specs[task_name].primary_metric
            key = f"{task_name}/{primary_name}"
            if key in metrics:
                primary_values.append(metrics[key])
        metrics["avg_metric"] = (
            float(sum(primary_values) / len(primary_values)) if primary_values else float("nan")
        )
        return metrics

    @torch.inference_mode()
    def val_epoch(self):
        self.model.eval()
        val_metrics = self._eval_epoch(self.val_loader, split="val")
        selection_key = self.cfg.val.selection_key
        selected = val_metrics[selection_key]
        if not math.isfinite(selected):
            self.logger.warn(f"val/{selection_key} is not finite; this epoch cannot be best.")

        improved = self.save_if_best(selected, metrics=val_metrics)
        early_stop = self.step_patience(improved)

        log_metrics = {f"val/{k}": v for k, v in val_metrics.items()} | {
            "val/avg": selected,
            "val/epoch": self.epoch,
        }
        self.logger.log_dict(log_metrics)
        self.push_logs()
        self.return_value = log_metrics | self.best_metrics
        return early_stop

    @torch.inference_mode()
    def test(self):
        if not hasattr(self, "test_loader") or len(self.test_loader) == 0:
            return
        if self.best_model is not None:
            self.model.load_state_dict(self.best_model)

        self.model.eval()
        test_metrics = self._eval_epoch(self.test_loader, split="test")
        log_metrics = {f"best/test/{k}": v for k, v in test_metrics.items()}
        self.logger.log_dict(log_metrics)
        self.push_logs()
        if hasattr(self, "return_value"):
            self.return_value |= log_metrics
        else:
            self.return_value = log_metrics

    # ------------------------------------------------------------
    # Reporting
    # ------------------------------------------------------------

    def get_best_pbar_metrics(self) -> dict:
        if hasattr(self, "return_value"):
            return {
                k.replace("val/", "").replace("best/test/", "test/"): v
                for k, v in self.return_value.items()
            }
        return {}

    def log_training_results(self):
        if not hasattr(self, "return_value"):
            return
        table = Table(title="Final Multitask Metrics", show_header=True)
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="green")
        for k, v in sorted(self.return_value.items()):
            table.add_row(k, f"{v:.4f}" if isinstance(v, float) else str(v))
        Console().print(table)
