import math
import multiprocessing as mp
from collections import defaultdict

import hydra
import numpy as np
import torch
import torch.nn as nn
from rich.console import Console
from rich.table import Table
from torch.nn import CrossEntropyLoss, MSELoss, PoissonNLLLoss
from torch.utils.data import DataLoader
from torch_brain.batching import collate
from torch_brain.samplers import RandomFixedWindowSampler, TrialSampler
from torch_brain.transforms import Compose

from core.nn.loss import WeightedMSELoss
from core.samplers import DistributedSamplerWrapper
from core.trainer import BaseTrainer
from core.utils.checkpoint import log_incompatible_keys
from core.utils.exceptions import TrainingConstraintsError
from core.utils.util import (
    get_num_params,
    get_num_trainable_params,
    human_readable,
    move_to_device,
)
from ibl_bwb_eval.tasks import DataType, get_ts1_readout_spec, get_ts1_supported_tasks
from pretrain.datasets import IBLBrainWideBenchMultiTaskBehavior

from .poyo_plus import POYOPlus


class POYOPlusMultitaskPretrain(BaseTrainer):
    """Supervised pretraining of POYO+ on every behavior at once, via its readout heads."""

    def setup(self, ckpt: dict | None):
        self.logger.info(f"Trainer: {self.__class__}")
        assert self.cfg.val.selection_key in ("avg_metric", "avg_loss"), (
            f"unsupported val.selection_key={self.cfg.val.selection_key}"
        )
        self._setup_tasks()
        self.setup_train_loader()
        self.setup_val_loader()

        self.setup_model()
        self.link_model(self.model)
        self._restore_model_from_ckpt(ckpt)

        self.model.to(self.device)
        self.model = self.make_ddp(
            self.model,
            find_unused_parameters=bool(
                self.cfg.get("ddp", {}).get("find_unused_parameters", False)
            ),
        )

        self.configure_optimizers()
        self.add_checkpoint_items(
            model=self.model,
            optimizer=self.optimizer,
            scheduler=self.scheduler,
        )

    def _restore_model_from_ckpt(self, ckpt: dict | None):
        if ckpt is None or "model_state_dict" not in ckpt:
            return

        strict = bool(self.cfg.ckpt.resume) and not bool(self.cfg.ckpt.model_only)
        state_dict = ckpt["model_state_dict"]
        try:
            if strict:
                self.model.load_state_dict(state_dict, strict=True)
                self.logger.info("Loaded model checkpoint with strict=True")
                return
        except RuntimeError as exc:
            self.logger.warn(f"Strict checkpoint load failed ({exc}). Retrying with strict=False.")

        log_incompatible_keys(self.model.load_state_dict(state_dict, strict=False), self.logger)
        self.logger.info("Loaded model checkpoint with strict=False")

    def _build_loss_fn(self, task_name: str):
        spec = self.readout_specs[task_name]
        if spec.data_type in {DataType.BINARY, DataType.MULTINOMIAL}:
            return CrossEntropyLoss()
        if spec.data_type == DataType.CONTINUOUS:
            return WeightedMSELoss() if spec.mask_key is not None else MSELoss()
        if spec.data_type == DataType.EVENT_RATE:
            return PoissonNLLLoss(log_input=True, full=True, eps=1e-9, reduction="mean")
        raise ValueError(f"Unsupported data type: {spec.data_type}")

    def _setup_tasks(self):
        tasks_cfg = self.cfg.get("tasks", None)
        if tasks_cfg is None:
            tasks = get_ts1_supported_tasks()
        elif isinstance(tasks_cfg, str):
            tasks = [tasks_cfg]
        else:
            try:
                tasks = list(tasks_cfg)
            except TypeError:
                tasks = [tasks_cfg]

        supported_tasks = set(get_ts1_supported_tasks())
        unknown_tasks = [task for task in tasks if task not in supported_tasks]
        if unknown_tasks:
            raise ValueError(
                f"Unsupported tasks: {unknown_tasks}. Supported tasks: {sorted(supported_tasks)}"
            )

        self.tasks = tasks
        self.readout_specs = {task: get_ts1_readout_spec(task) for task in self.tasks}
        self.loss_fns = {task: self._build_loss_fn(task) for task in self.tasks}
        task_loss_weights_cfg = self.cfg.get("multitask", {}).get("task_loss_weights", {})
        self.task_loss_weights = {
            task: float(task_loss_weights_cfg.get(task, 1.0)) for task in self.tasks
        }
        self.logger.info(f"Multitask pretraining tasks: {self.tasks}")
        self.logger.info(f"Multitask task_loss_weights: {self.task_loss_weights}")

    def setup_model(self):
        self.model = hydra.utils.instantiate(self.cfg.model)
        if not isinstance(self.model, POYOPlus):
            raise TypeError(
                f"{type(self).__name__} requires a POYOPlus model, got {type(self.model).__name__}"
            )
        self.logger.info(f"Precision: {self.precision}")
        self.logger.info(f"Model: {self.model.__class__}")
        self.model.train()

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
                f"Batch size {self.cfg.batch_size // self.world_size} is larger than dataset size {len(self.train_sampler)}. "
                "No batches would be produced with drop_last=True."
            )

        has_workers = self.cfg.num_workers > 0
        self.train_loader = DataLoader(
            self.train_dataset,
            sampler=self.train_sampler,
            collate_fn=collate,
            batch_size=self.cfg.batch_size // self.world_size,
            num_workers=self.cfg.num_workers,
            drop_last=True,
            pin_memory=self.cfg.pin_memory,
            prefetch_factor=self.cfg.prefetch_factor if has_workers else None,
            persistent_workers=self.cfg.persistent_workers if has_workers else False,
            multiprocessing_context=mp.get_context("fork") if has_workers else None,
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

        has_workers = self.cfg.num_workers > 0
        self.val_loader = DataLoader(
            self.val_dataset,
            sampler=self.val_sampler,
            collate_fn=collate,
            batch_size=self.cfg.batch_size // self.world_size,
            num_workers=self.cfg.num_workers,
            pin_memory=self.cfg.pin_memory,
            prefetch_factor=self.cfg.prefetch_factor if has_workers else None,
            persistent_workers=self.cfg.persistent_workers if has_workers else False,
            multiprocessing_context=mp.get_context("fork") if has_workers else None,
        )
        self.logger.info(
            f"Validating on {len(self.val_loader)} batches, total {len(self.val_sampler)} samples"
        )

    def link_model(self, model: nn.Module):
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
        self.logger.info(f"Number of parameters: {human_readable(num_params)} ({num_params:,})")
        self.logger.info(
            f"Number of trainable parameters: {human_readable(num_trainable_params)} ({num_trainable_params:,})"
        )
        if skipped_params:
            self.logger.info(f"Skipped lazy parameters: {skipped_params}")

    def configure_optimizers(self):
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

    def predict(self, batch):
        return self.model(**batch["model_inputs"], return_dict=True)

    @staticmethod
    def _unwrap_collated_value(x):
        # `chain(...)` fields can be collated into `(tensor, metadata)`.
        if isinstance(x, tuple):
            return x[0]
        return x

    def _task_loss(self, task_name: str, pred: torch.Tensor, batch: dict):
        if task_name not in batch.get("target_values", {}):
            return None

        spec = self.readout_specs[task_name]
        target = self._unwrap_collated_value(batch["target_values"][task_name])

        if spec.data_type in {DataType.BINARY, DataType.MULTINOMIAL}:
            target = target.squeeze(-1).long()
            return self.loss_fns[task_name](pred, target)

        target = target.view(-1, spec.dim).float()

        if spec.data_type == DataType.CONTINUOUS and task_name in batch.get("target_masks", {}):
            mask = self._unwrap_collated_value(batch["target_masks"][task_name]).bool()
            if mask.ndim == 1:
                mask = mask.unsqueeze(-1).expand(-1, spec.dim)
            mask = mask.reshape(-1, spec.dim).float()
            if mask.sum() <= 0:
                return None
            return self.loss_fns[task_name](pred, target, mask)

        if spec.data_type == DataType.EVENT_RATE:
            return self.loss_fns[task_name](pred, target)

        return self.loss_fns[task_name](pred, target)

    def _build_eval_metrics(self):
        eval_metrics = {}
        for task_name in self.tasks:
            eval_metrics[task_name] = {
                metric_name: metric_ctor().to(self.device)
                for metric_name, metric_ctor in self.readout_specs[task_name].metrics.items()
            }
        return eval_metrics

    def _prepare_metric_inputs(self, task_name: str, pred: torch.Tensor, batch: dict):
        if task_name not in batch.get("target_values", {}):
            return None, None

        spec = self.readout_specs[task_name]
        target = self._unwrap_collated_value(batch["target_values"][task_name])

        if spec.data_type in {DataType.BINARY, DataType.MULTINOMIAL}:
            target = target.squeeze(-1).long()
            if target.numel() == 0:
                return None, None
            return pred, target

        pred = pred.view(-1, spec.dim)
        target = target.view(-1, spec.dim).float()

        if spec.data_type == DataType.CONTINUOUS and task_name in batch.get("target_masks", {}):
            mask = self._unwrap_collated_value(batch["target_masks"][task_name]).bool()
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
            if v.ndim > 0:
                return v.mean().item()
            return v.item()
        return float(v)

    def _get_task_loss_weight(self, task_name: str) -> float:
        if not hasattr(self, "task_loss_weights"):
            return 1.0
        return float(self.task_loss_weights.get(task_name, 1.0))

    def compute_multitask_loss(self, pred_dict: dict[str, torch.Tensor], batch: dict):
        task_losses = {}
        weighted_task_losses = {}
        for task_name, pred in pred_dict.items():
            loss = self._task_loss(task_name, pred, batch)
            if loss is None:
                continue
            task_losses[task_name] = loss
            weighted_task_losses[task_name] = loss * self._get_task_loss_weight(task_name)

        if len(task_losses) == 0:
            return None, {}, {}

        total_loss = torch.stack(list(weighted_task_losses.values())).mean()
        return total_loss, task_losses, weighted_task_losses

    def train_epoch(self):
        loader = self.train_loader
        if self.cfg.log_train_step:
            loader = self.logger.get_pbar(loader, prefix="train")

        epoch_metrics = defaultdict(list)

        for batch in loader:
            batch = move_to_device(batch, device=self.device)
            self.optimizer.zero_grad()
            with torch.autocast(device_type="cuda", dtype=self.precision.dtype):
                pred_dict = self.predict(batch)
                total_loss, task_losses, weighted_task_losses = self.compute_multitask_loss(
                    pred_dict, batch
                )

            if total_loss is None:
                continue

            total_loss.backward()

            self.clip_and_log_grad_norm()
            self.optimizer.step()
            self.scheduler.step()
            self.train_step += 1

            step_metrics = {"train/loss": total_loss.item()}
            for task_name, loss in task_losses.items():
                step_metrics[f"train/{task_name}_loss"] = loss.item()
                step_metrics[f"train/{task_name}_loss_weighted"] = weighted_task_losses[
                    task_name
                ].item()
                step_metrics[f"train/{task_name}_loss_weight"] = self._get_task_loss_weight(
                    task_name
                )

            if self.cfg.log_train_step:
                self.logger.log("train/loss", step_metrics["train/loss"], pbar=True, wandb=False)
                self.logger.log_dict(step_metrics)
                self.logger.log("train/lr", self.scheduler.get_last_lr()[0], pbar=False)
                self.push_logs()
            else:
                for key, value in step_metrics.items():
                    epoch_metrics[key].append(value)

        if epoch_metrics:
            # log_train_step off: the buffer is only flushed at val, so a per-step log
            # would push the last batch's value as if it were the epoch's.
            self.logger.log_dict({k: sum(v) / len(v) for k, v in epoch_metrics.items()})
            self.logger.log("train/lr", self.scheduler.get_last_lr()[0], pbar=False)

    @torch.inference_mode()
    def _eval_epoch(self, loader: DataLoader, split: str):
        loss_sums: dict[str, float] = dict.fromkeys(self.tasks, 0.0)
        loss_counts: dict[str, int] = dict.fromkeys(self.tasks, 0)
        avg_losses = []
        eval_metrics = self._build_eval_metrics()
        valid_metric_tasks = set()

        for batch in self.logger.get_pbar(loader, prefix=split):
            batch = move_to_device(batch, device=self.device)
            with torch.autocast(device_type="cuda", dtype=self.precision.dtype):
                pred_dict = self.predict(batch)
                total_loss, task_losses, _ = self.compute_multitask_loss(pred_dict, batch)
            if total_loss is None:
                continue
            avg_losses.append(total_loss.item())
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
        metrics = {f"{task}_loss": v for task, v in pooled.items() if not math.isnan(v)}
        for task_name in self.tasks:
            if task_name not in valid_metric_tasks:
                continue
            for metric_name, metric in eval_metrics[task_name].items():
                metrics[f"{task_name}/{metric_name}"] = self._to_scalar_metric(metric.compute())

        avg_loss = self.reduce_mean(sum(avg_losses), len(avg_losses))
        metrics["avg_loss"] = float("inf") if math.isnan(avg_loss) else avg_loss
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
        val_metrics = self._eval_epoch(self.val_loader, split="val")
        selection_key = self.cfg.val.selection_key
        selected = val_metrics[selection_key]
        if not math.isfinite(selected):
            self.logger.warn(f"val/{selection_key} is not finite; this epoch cannot be best.")

        # best-model selection only: this trainer runs its full epoch budget
        self.save_if_best(selected, metrics=val_metrics)

        log_metrics = {f"val/{k}": v for k, v in val_metrics.items()} | {
            "val/avg": selected,
            "val/epoch": self.epoch,
        }
        self.logger.log_dict(log_metrics)
        self.push_logs()
        self.return_value = log_metrics

    @torch.inference_mode()
    def test(self):
        if not hasattr(self, "test_loader"):
            return
        if self.best_model is not None:
            self.model.load_state_dict(self.best_model)

        test_metrics = self._eval_epoch(self.test_loader, split="test")
        log_metrics = {f"best/test/{k}": v for k, v in test_metrics.items()}
        self.logger.log_dict(log_metrics)
        self.push_logs()
        if hasattr(self, "return_value"):
            self.return_value |= log_metrics
        else:
            self.return_value = log_metrics

    def log_training_results(self):
        if not hasattr(self, "return_value"):
            return
        table = Table(title="Final Multitask Metrics", show_header=True)
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="green")
        for k, v in sorted(self.return_value.items()):
            table.add_row(k, f"{v:.4f}" if isinstance(v, float) else str(v))
        Console().print(table)
