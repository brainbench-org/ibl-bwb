import multiprocessing as mp

import hydra
import torch
from omegaconf import DictConfig, OmegaConf
from rich.console import Console
from rich.table import Table
from torch.utils.data import DataLoader
from torch_brain.batching import collate
from torch_brain.samplers import RandomFixedWindowSampler, SequentialFixedWindowSampler
from torch_brain.transforms import Compose

from core.model import BaseModel
from core.trainer import BaseTrainer
from core.utils.exceptions import TrainingConstraintsError
from core.utils.util import (
    get_num_params,
    get_num_trainable_params,
    human_readable,
    move_to_device,
)
from ibl_bwb_eval.metrics import aggregate_metrics
from ibl_bwb_eval.tasks import TS2Task, get_ts2_readout_spec
from ts2.ts2_dataset import IBLBrainWideBenchTS2
from ts2.ts2_test_mixin import TS2TestMixin, select_scored_entries


class TS2EvalTrainer(TS2TestMixin, BaseTrainer):
    """Standardized evaluation Trainer for Task Suite 2 (neural prediction).

    This class defines the base Trainer for benchmark neural prediction tasks. This interface
    abstracts away the common boilerplate code for evaluating models on the benchmark, and allows
    for easy customization of the training loop. Both tasks reconstruct held-out spike counts
    stripped from the input before ``input_fn``, and only those entries are scored. Several
    methods related to standardized evaluation are declared final and should not be overridden.
    Please refer to :class:`~ts2.models.single_session.lfads.LFADSEvalTrainer` for a template for
    customizing the training logic.

    Available to customize (override these methods in a custom Trainer, if needed):

    - setup: Set up the trainer. This includes setting up the tasks, data loaders, model, optimizer, and scheduler.
    - setup_model: Instantiate and initialize the model.
    - link_model: Link the datasets to the model.
    - setup_finetuning: Set up the finetuning strategy.
    - setup_optimizers: Configure the optimizer (and scheduler, if applicable).
    - loss: Define the loss function.
    - train_epoch: Define the training loop. This is flexible and allows for custom training logic, such as
      adding gradient accumulation, curriculum learning, etc.
    - training_step: Compute the loss for one batch. This is where a model that corrupts its own
      input applies its masker.
    - predict: Given a batch of data, including inputs as defined by the model's input_fn and any
      masking information, generate the model's predictions.

    See :doc:`/guides/ts2` for more details on the interface.
    """

    def __init__(self, cfg: DictConfig, rank: int, world_size: int):
        """Initialize the TS2EvalTrainer.

        Calls the base :class:`~core.trainer.BaseTrainer` constructor then sets up
        early stopping state from the config.

        Args:
            cfg: Hydra configuration object.
            rank: Local process rank (device index).
            world_size: Total number of processes in the distributed group.
        """
        super().__init__(cfg, rank, world_size)

        self.return_value = {}

    def setup(self, ckpt: dict | None):
        """Set up all trainer components.

        Called automatically by the base constructor. Initializes tasks, data loaders,
        model, DDP wrapping, optimizer, and finetuning strategy in order.

        Args:
            ckpt: Checkpoint dictionary loaded from disk, or ``None`` for a fresh run.
        """
        self.logger.info(f"Trainer: {self.__class__}")

        if ckpt is None:
            assert not self.cfg.finetuning.enable, (
                "No checkpoint provided but finetuning is enabled"
            )

        # tasks
        self._setup_tasks()

        # data loaders
        self.setup_train_loader()
        self.setup_val_loader()
        self.setup_test_loader()

        # model
        self.setup_model(ckpt)
        self.link_model(self.model, ckpt)

        # eval-only runs score the checkpoint; finetuning paths load their own weights in link_model
        if self.cfg.num_epochs == 0 and not self.cfg.finetuning.enable:
            if ckpt is not None:
                self.model.load_state_dict(ckpt["model_state_dict"])
                self.logger.info("Loaded model weights from checkpoint for eval")
            else:
                self.logger.warn(
                    "num_epochs=0 without a checkpoint: evaluating the model as initialized"
                )

        # move model to device after linking, in case new parameters were created
        self.model.to(self.device)
        self.model = self.make_ddp(self.model)

        # optimizer and scheduler (num_epochs=0 is an eval-only run, nothing to optimize)
        if self.cfg.num_epochs > 0:
            self.setup_optimizers(ckpt)
        else:
            self.optimizer = self.scheduler = None
            self.logger.info("num_epochs=0: eval only, skipping optimizer setup")

        # finetuning
        self.setup_finetuning()

    def setup_model(self, ckpt: dict | None):
        """Instantiate and register the model.

        Instantiates the model from the Hydra config and registers it for checkpointing.
        Override this to load pretrained weights or modify the model architecture.

        Args:
            ckpt: Checkpoint dictionary, available for loading pretrained weights.
        """
        self.model = hydra.utils.instantiate(self.cfg.model)
        if not isinstance(self.model, BaseModel):
            raise TypeError(
                f"{type(self).__name__} requires a BaseModel subclass, got {type(self.model).__name__}"
            )
        self.add_checkpoint_items(model=self.model)

        self.logger.info(f"Precision: {self.precision}")
        self.logger.info(f"Model: {self.model.__class__}")
        self.model.train()

    def link_model(self, model: BaseModel, ckpt: dict | None):
        """Link datasets to the model.

        Attaches the configured transforms and the model input_fn to the train/val/test
        dataset transform pipelines, then calls ``model.link_datasets`` to register the
        datasets with the model.

        Args:
            model: The instantiated model to link.
            ckpt: Checkpoint dictionary, passed through for subclass use.
        """
        # attach model transforms to end of dataset transform pipelines
        for split in ["train", "val", "test"]:
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
            self.test_loader.dataset,
        )

        num_params, skipped_params = get_num_params(self.model)
        num_trainable_params = get_num_trainable_params(self.model, return_skipped=False)
        self.logger.info(f"Number of parameters: {human_readable(num_params)} ({num_params:,})")
        self.logger.info(
            f"Number of trainable parameters: {human_readable(num_trainable_params)} ({num_trainable_params:,})"
        )
        if skipped_params:
            self.logger.info(f"Skipped lazy parameters: {skipped_params}")

    def setup_finetuning(self):
        """Set up the finetuning strategy from config.

        Instantiates and initializes the finetuning strategy if ``finetuning.strategy``
        is specified in the config. Sets ``self.ft_strategy`` to ``None`` otherwise.
        """
        self.ft_strategy = None
        if self.cfg.finetuning.strategy is None:
            self.logger.info("No finetuning strategy specified, skipping")
            return
        self.ft_strategy = hydra.utils.instantiate(
            self.cfg.finetuning.strategy,
            model=self.model,
            cfg=self.cfg,
            _recursive_=False,
        )
        if self.ft_strategy.enable:
            self.ft_strategy.setup()

    def loss(self, pred, target, mask=None):
        """Compute the reconstruction loss.

        Args:
            pred: Model predictions.
            target: Ground truth targets.
            mask: Optional boolean mask. If provided, loss is averaged only over
                masked positions.

        Returns:
            torch.Tensor: Scalar loss value.
        """
        if mask is None:
            return self.loss_fn(pred, target).mean()

        return self.loss_fn(pred, target)[mask].mean()

    def train_epoch(self):
        """Run one training epoch.

        Iterates over the train loader, computes predictions and loss, and updates
        model parameters via optimizer and scheduler. Logs loss per step if
        ``log_train_step`` is enabled, otherwise logs the epoch average.
        """
        self.model.train()
        if self.ft_strategy is not None and self.ft_strategy.enable:
            self.ft_strategy.update(self.epoch)

        loader = self.train_loader
        if self.cfg.log_train_step:
            loader = self.logger.get_pbar(loader, prefix="train")

        epoch_losses = []
        for X, y in loader:
            X = move_to_device(X, device=self.device)
            y = move_to_device(y, device=self.device)
            self.optimizer.zero_grad()
            with torch.autocast(device_type=self.device.type, dtype=self.precision.dtype):
                loss = self.training_step(X, y)

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
            self.logger.log("train/lr", self.scheduler.get_last_lr()[0], pbar=False)

    def predict(self, X, mask=None, mask_timestamps=None, mask_units=None):
        """Generate model predictions for a batch.

        Override this if the model requires masking information at inference time
        (e.g. for masked autoencoder models).

        Args:
            X: Input batch dict containing ``model_inputs``.
            mask: Optional boolean mask for masked prediction.
            mask_timestamps: Optional timestamps for masked positions.
            mask_units: Optional unit indices for masked positions.

        Returns:
            torch.Tensor: Model output predictions.
        """
        return self.model(**X["model_inputs"])

    def training_step(self, X: dict, y: dict) -> torch.Tensor:
        """Compute loss for one training batch.

        Override this to customize masking, auxiliary losses, or any
        model-specific training logic. The base implementation is a
        plain forward pass against the benchmark-fixed target.

        Args:
            X: Input batch dict containing ``model_inputs``.
            y: Target batch dict containing ``values`` (benchmark-fixed spike counts).

        Returns:
            torch.Tensor: Scalar loss.
        """
        return self.loss(self.predict(X), y["values"])

    def setup_optimizers(self, ckpt: dict | None):
        """Configure the AdamW optimizer and OneCycleLR scheduler.

        Weight decay skips 1-D params and the names in ``cfg.no_weight_decay``.
        Override this to use a different optimizer or scheduler.

        Args:
            ckpt: Checkpoint dictionary. Optimizer/scheduler resumption is not yet supported.

        Returns:
            tuple: ``(optimizer, scheduler)``
        """
        if self.model is None:
            raise ValueError("Trying to configure optimizers before the model is setup")
        if self.train_loader is None:
            raise ValueError("Trying to configure optimizers before the train data loader is setup")

        grouped_parameters = self.get_param_groups()

        self.optimizer = torch.optim.AdamW(
            grouped_parameters,
            lr=self.cfg.base_lr,
        )
        self.scheduler = torch.optim.lr_scheduler.OneCycleLR(
            self.optimizer,
            max_lr=self.cfg.base_lr,
            epochs=self.cfg.num_epochs,
            steps_per_epoch=len(self.train_loader),
            pct_start=self.cfg.pct_start,
            anneal_strategy="cos",
            div_factor=self.cfg.div_factor,
            final_div_factor=1e4,
        )
        self.add_checkpoint_items(optimizer=self.optimizer, scheduler=self.scheduler)
        return self.optimizer, self.scheduler

    def _setup_tasks(self):
        from typing import get_args

        from torch.nn import PoissonNLLLoss

        task = self.cfg.task
        assert task in get_args(TS2Task), (
            f"Invalid task '{task}', must be one of {get_args(TS2Task)}"
        )
        self.task = task
        self.logger.info(f"Task: {self.task}")
        self.loss_fn = PoissonNLLLoss(log_input=True, full=True, eps=1e-9, reduction="none")

    # ------------------------------------------------------------
    # Below are the standard methods that should not be overridden
    # ------------------------------------------------------------

    def setup_train_loader(self):
        self.train_dataset = IBLBrainWideBenchTS2(
            root=self.cfg.data_root,
            split="train",
            recording_id=self.cfg.recording_id,
            task=self.task,
            context_length=OmegaConf.select(self.cfg, "context_length", default=1.0),
        )

        self.train_sampler = RandomFixedWindowSampler(
            sampling_intervals=self.train_dataset.get_sampling_intervals(),
            window_length=self.train_dataset.TARGET_WINDOW,
            generator=torch.Generator().manual_seed(self.cfg.seed),
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
            batch_size=self.cfg.batch_size,
            num_workers=self.cfg.num_workers,
            drop_last=True,
            pin_memory=self.cfg.pin_memory,
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
        self.val_dataset = IBLBrainWideBenchTS2(
            root=self.cfg.data_root,
            split="val",
            recording_id=self.cfg.recording_id,
            task=self.task,
            mask_input=self.cfg.val.mask_input,
            context_length=OmegaConf.select(self.cfg, "context_length", default=1.0),
        )
        self.val_sampler = SequentialFixedWindowSampler(
            sampling_intervals=self.val_dataset.get_sampling_intervals(),
            window_length=self.val_dataset.TARGET_WINDOW,
            step=self.train_dataset.BIN_SIZE * self.cfg.val.step_factor,
        )
        has_workers = self.cfg.num_workers > 0
        self.val_loader = DataLoader(
            self.val_dataset,
            sampler=self.val_sampler,
            collate_fn=collate,
            batch_size=self.cfg.batch_size,
            num_workers=self.cfg.num_workers,
            pin_memory=self.cfg.pin_memory,
            persistent_workers=self.cfg.persistent_workers if has_workers else False,
            multiprocessing_context=mp.get_context("fork") if has_workers else None,
        )
        self.logger.info(
            f"Val dataset: {len(self.val_dataset.get_session_ids())} sessions, {len(self.val_dataset.get_unit_ids())} units"
        )
        self.logger.info(
            f"Validating on {len(self.val_loader)} batches, total {len(self.val_sampler)} samples"
        )
        if not self.val_dataset.mask_input:
            self.logger.warn(
                "val.mask_input=false: the val input keeps the held-out spikes, so the val "
                "metrics leak and are not comparable to test"
            )

    def log_training_results(self):
        """Print a Rich table summarizing the best metrics at the end of training."""
        if self.best_metrics:
            table = Table(title="Final Best Metrics", show_header=True)
            table.add_column("Metric", style="cyan")
            table.add_column("Value", style="green")
            for k, v in sorted(self.best_metrics.items(), key=lambda item: item[0]):
                table.add_row(k, f"{v:.4f}" if isinstance(v, float) else str(v))
            Console().print(table)

    def _val_epoch(self, loader: DataLoader) -> dict:
        """Score val exactly as ``test`` does.

        The held-out entries are stripped from the input in the dataset, so we forward
        the masked input and score only the ``y["mask"]`` entries. This keeps the val
        metric faithful to the test metric.

        ``val.mask_input=false`` leaves the input intact, so the metric leaks.
        """
        self.model.eval()
        spec = get_ts2_readout_spec(self.task)
        metrics = {name: metric().to(self.device) for name, metric in spec.metrics.items()}

        for X, y in self.logger.get_pbar(loader, prefix="val"):
            X = move_to_device(X, device=self.device)
            y = move_to_device(y, device=self.device)

            with torch.autocast(device_type=self.device.type, dtype=self.precision.dtype):
                target = y["values"]
                mask = y["mask"]

                if mask.sum() == 0:
                    continue

                assert mask.dtype == torch.bool, "Mask must be a boolean tensor"

                pred = self.predict(X, mask, y.get("mask_timestamps"), y.get("mask_units"))
                pred, target, _, _ = select_scored_entries(pred, target, mask, spec.mask_dim)

                for metric in metrics.values():
                    metric.update(pred, target)

        return aggregate_metrics(metrics)

    @torch.inference_mode()
    def val_epoch(self):
        spec = get_ts2_readout_spec(self.task)
        val_metrics = self._val_epoch(self.val_loader)
        self.logger.update_epoch_postfix(self.get_best_pbar_metrics())

        improved = self.save_if_best(val_metrics[spec.primary_metric], metrics=val_metrics)
        early_stop = self.step_patience(improved)

        val_metrics = (
            {f"val/{k}": v for k, v in val_metrics.items()}
            | {"val/avg": val_metrics[spec.primary_metric]}
            | {"val/epoch": self.epoch}
        )
        self.logger.log_dict(val_metrics)
        self.push_logs()

        self.return_value = val_metrics | self.best_metrics
        return early_stop
