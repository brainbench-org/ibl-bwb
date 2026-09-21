import math
import multiprocessing as mp

import hydra
import numpy as np
import torch
import torch.nn as nn
from omegaconf import OmegaConf
from rich.console import Console
from rich.table import Table
from torch.utils.data import DataLoader
from torch_brain.batching import collate
from torch_brain.samplers import RandomFixedWindowSampler, TrialSampler
from torch_brain.transforms.container import Compose

from core.samplers import DistributedSamplerWrapper
from core.trainer import BaseTrainer
from core.utils.exceptions import TrainingConstraintsError
from core.utils.util import get_num_params, get_num_trainable_params, human_readable, move_to_device
from ibl_bwb_eval.tasks import DataType, TargetResolution, get_ts1_readout_spec
from pretrain.datasets import IBLBrainWideBenchMultiTaskBehavior

from .masker import NEDSMasker
from .neds import NEDS


class NEDSPretrain(BaseTrainer):
    """Multimodal masked modelling for NEDS, over spikes and the behaviors alike."""

    def setup(self, ckpt: dict | None):
        self.logger.info(f"Trainer: {self.__class__}")
        self.reset_best_tracking(minimize=True)

        # ssl task
        self._setup_ssl_task()

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

    def _setup_ssl_task(self):
        self.logger.info("Task: Multi Modal Mask SSL")

        self.tasks = list(self.cfg.tasks)
        self.modalities = ["spikes", *self.tasks]

        specs = {task: get_ts1_readout_spec(task) for task in self.tasks}
        # cached, as loss() reads it per modality per batch and building a spec is not free
        self.sequence_modalities = {
            task
            for task, spec in specs.items()
            if spec.target_resolution == TargetResolution.SEQUENCE
        }

        self.loss_fn_dic = {"spikes": nn.PoissonNLLLoss(reduction="none", log_input=True)}
        for task, spec in specs.items():
            self.loss_fn_dic[task] = self._build_loss_fn(spec)

        cfg_weights = self.cfg.get("modality_weights", None)
        if cfg_weights is None:
            cfg_weights = {}
        elif OmegaConf.is_config(cfg_weights):
            cfg_weights = OmegaConf.to_container(cfg_weights, resolve=True)
        self.modality_weights = {mod: float(cfg_weights.get(mod, 1.0)) for mod in self.modalities}
        self.logger.info(
            "Modality weights: " + ", ".join(f"{m}={w:g}" for m, w in self.modality_weights.items())
        )

    def _build_loss_fn(self, spec):
        if spec.data_type in {DataType.BINARY, DataType.MULTINOMIAL}:
            return nn.CrossEntropyLoss(reduction="none")
        if spec.data_type == DataType.CONTINUOUS:
            return nn.MSELoss(reduction="none")
        if spec.data_type == DataType.EVENT_RATE:
            return nn.PoissonNLLLoss(reduction="none", log_input=True)
        raise ValueError(f"Unsupported data type: {spec.data_type}")

    def predict(self, model_inputs, keep_masks, modality_masks):
        return self.model(model_inputs, keep_masks, modality_masks)

    def loss(self, preds, targets, keep_masks, masks, modality):
        loss_fn = self.loss_fn_dic[modality]
        pred = preds[modality]  # (B, 1, 1) | (B, T, 1) | (B, T, N)
        target = targets[modality]  # (B, 1, 1) | (B, T, 1) | (B, T, N)
        keep_mask = keep_masks[modality]  # (B, T, 1) | (B, T, N)
        mask = masks[modality]  # (B, T, 1)

        if modality in self.sequence_modalities:
            pred = pred.squeeze(1)  # (B, 1, D) -> (B, D)
            target = target.to(torch.long).clip(min=0)
            target = target[:, 0, 0]  # (B, 1, 1) -> (B,)
            keep_mask = keep_mask[:, 0, 0]  # (B, T, 1) -> (B,)
            mask = mask[:, 0, 0]  # (B, T, 1) -> (B,)

        valid = keep_mask & mask

        count = valid.sum()
        if count == 0:
            return pred.new_tensor(0.0), count

        loss = loss_fn(pred, target)[valid].mean()
        return loss, count

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

        if len(self.train_sampler) < self.cfg.batch_size:
            raise TrainingConstraintsError(
                f"Batch size {self.cfg.batch_size} is larger than dataset size {len(self.train_sampler)}. "
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
            f"Train dataset: {len(self.train_dataset.get_session_ids())} sessions, {len(self.train_dataset.get_unit_ids())} units"
        )
        self.logger.info(
            f"Training on {len(self.train_loader)} batches, total {len(self.train_loader) * self.cfg.batch_size} samples"
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

    def train_epoch(self):
        self.model.train()
        loader = self.train_loader
        if self.cfg.log_train_step:
            loader = self.logger.get_pbar(loader, prefix="train")

        for X in loader:
            X = move_to_device(X, device=self.device)
            self.optimizer.zero_grad()
            with torch.autocast(device_type="cuda", dtype=self.precision.dtype):
                inputs = X["model_inputs"]
                keep_masks = X["keep_masks"]

                targets = {modality: inputs[modality].clone() for modality in self.modalities}

                _spikes = inputs["spikes"]
                modality_masks = self.masker(_spikes)

                preds = self.predict(inputs, keep_masks, modality_masks)

                # a mask scheme zeroes out whole modality groups, so normalizing by the
                # modality count would scale the gradient by whichever scheme was drawn
                total_loss, total_weight = 0.0, 0.0
                modality_losses = {}
                for modality in self.modalities:
                    loss, count = self.loss(preds, targets, keep_masks, modality_masks, modality)
                    modality_losses[modality] = loss.item()
                    if count == 0:
                        continue
                    total_loss += self.modality_weights[modality] * loss
                    total_weight += self.modality_weights[modality]

                if total_weight == 0:
                    raise TrainingConstraintsError(
                        "No modality had a masked target in this batch, so there is nothing "
                        "to predict. Check masker.mask_ratio and masker.mask_types."
                    )

                total_loss = total_loss / total_weight

            total_loss.backward()
            self.clip_and_log_grad_norm()
            self.optimizer.step()
            self.scheduler.step()
            self.train_step += 1

            if self.cfg.log_train_step:
                self.logger.log("train/loss", total_loss.item(), pbar=True)
                for modality, loss_val in modality_losses.items():
                    self.logger.log(f"train/{modality}_loss", loss_val)
                self.logger.log("train/lr", self.scheduler.get_last_lr()[0], pbar=False)

                self.push_logs()

    @torch.inference_mode()
    def val_epoch(self):
        self.model.eval()

        val_tracker = {mod: {"sum": 0.0, "count": 0} for mod in self.modalities}
        generator = torch.Generator(device=self.device).manual_seed(self.cfg.seed)
        for X in self.val_loader:
            X = move_to_device(X, device=self.device)
            with torch.autocast(device_type="cuda", dtype=self.precision.dtype):
                inputs = X["model_inputs"]
                keep_masks = X["keep_masks"]

                targets = {modality: inputs[modality].clone() for modality in self.modalities}

                _spikes = inputs["spikes"]
                modality_masks = self.masker(_spikes, generator=generator)

                preds = self.predict(inputs, keep_masks, modality_masks)

                for modality in self.modalities:
                    loss, count = self.loss(preds, targets, keep_masks, modality_masks, modality)

                    val_tracker[modality]["sum"] += loss.item() * count.item()
                    val_tracker[modality]["count"] += count.item()

        metrics = {}
        modality_means = []

        for modality in self.modalities:
            m_sum = val_tracker[modality]["sum"]
            m_count = val_tracker[modality]["count"]

            # every rank calls this once per modality, in this order
            mean = self.reduce_mean(m_sum, m_count)
            if math.isnan(mean):
                continue

            metrics[f"{modality}_loss"] = mean
            modality_means.append(mean)

        # a target count runs from B*T for a behavior to B*T*N for spikes, so pooling the
        # counts would select on spike reconstruction alone
        val_loss = sum(modality_means) / len(modality_means) if modality_means else float("inf")
        metrics["loss"] = val_loss

        self.save_if_best(val_loss, metrics=metrics)

        val_metrics = {f"val/{k}": v for k, v in metrics.items()} | {
            "val/avg": val_loss,
            "val/epoch": self.epoch,
        }
        self.logger.log_dict(val_metrics)
        self.push_logs()

        self.return_value = val_metrics

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

    def setup_model(self, ckpt: dict | None = None):
        self.model = hydra.utils.instantiate(self.cfg.model)
        if not isinstance(self.model, NEDS):
            raise TypeError(
                f"{type(self).__name__} requires a NEDS model, got {type(self.model).__name__}"
            )
        self.model.modalities = self.modalities
        self.masker = hydra.utils.instantiate(self.cfg.masker, modalities=self.modalities)
        if not isinstance(self.masker, NEDSMasker):
            raise TypeError(
                f"{type(self).__name__} requires a NEDSMasker masker, got {type(self.masker).__name__}"
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

        num_params, skipped_params = get_num_params(self.model)
        num_trainable_params = get_num_trainable_params(self.model, return_skipped=False)
        num_backbone_params = get_num_params(
            self.model.encoder, return_skipped=False
        ) + get_num_params(self.model.encoder_norm, return_skipped=False)

        self.logger.info(f"Number of parameters: {human_readable(num_params)} ({num_params:,})")
        self.logger.info(
            f"Number of trainable parameters: {human_readable(num_trainable_params)} ({num_trainable_params:,})"
        )
        self.logger.info(
            f"Number of backbone parameters (excl. embeddings & stitchers): {human_readable(num_backbone_params)} ({num_backbone_params:,})"
        )
        if skipped_params:
            self.logger.info(f"Skipped lazy parameters: {skipped_params}")

    def log_training_results(self):
        table = Table(title="Final Best Metrics", show_header=True)
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="green")
        table.add_row("best/epoch", str(self.best_epoch))
        table.add_row("best/val/avg", f"{self.best_val:.4f}")
        Console().print(table)
