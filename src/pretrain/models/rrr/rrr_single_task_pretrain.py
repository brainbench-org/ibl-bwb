"""Single-behavior pretraining for the reduced-rank decoder.

``V`` is learned jointly across the pretrain sessions while each keeps its own
``U``/``b``. ``V``'s last axis is the task's output dim, so each behavior needs its
own run and checkpoint. Selection is on the task's primary metric pooled over
sessions; ``best.pt`` is the entire output of this phase.
"""

import multiprocessing as mp
from typing import Literal

import hydra
import numpy as np
import torch
from torch.nn import CrossEntropyLoss, MSELoss, PoissonNLLLoss
from torch.utils.data import DataLoader
from torch_brain.samplers import TrialSampler
from torch_brain.transforms import Compose

from core.batching.collate import supervised_collate
from core.nn.loss import WeightedMSELoss
from core.samplers import DistributedSamplerWrapper, SessionBatchSampler
from core.trainer import BaseTrainer
from core.utils.util import get_num_params, human_readable, move_to_device
from ibl_bwb_eval.metrics import aggregate_metrics
from ibl_bwb_eval.tasks import DataType, TargetResolution, get_ts1_readout_spec
from pretrain.datasets import IBLBrainWideBenchSingleTaskBehavior

from .rrr import RRRDecoder


class RRRSingleTaskPretrain(BaseTrainer):
    """Supervised pretraining of the shared basis ``V`` on one behavior."""

    def setup(self, ckpt: dict | None):
        self.logger.info(f"Trainer: {self.__class__}")

        if self.cfg.test.get("enable", False):
            raise ValueError(
                "This phase builds no test loader: the benchmark numbers come from the "
                "eval trainer on an eval session. Set test.enable=false (the "
                "rrr_pretrain config does)."
            )

        self.setup_tasks()
        self._setup_pooled_loader("train")
        self._setup_pooled_loader("val")

        self.setup_model(ckpt)
        self.link_model(self.model)

        # move model to device after linking, in case new parameters were created
        self.model.to(self.device)
        # Each step touches one session's U/b and leaves the others untouched.
        self.model = self.make_ddp(self.model, find_unused_parameters=True)

        self.setup_optimizers(ckpt)

        self.reset_best_tracking()

    def setup_tasks(self):
        self.task = self.cfg.task
        self.readout_spec = get_ts1_readout_spec(self.task)

        self.logger.info(f"Task: {self.task} (supervised pretraining)")

        if self.readout_spec.data_type in (DataType.BINARY, DataType.MULTINOMIAL):
            self.loss_fn = CrossEntropyLoss()
        elif self.readout_spec.data_type == DataType.CONTINUOUS:
            self.loss_fn = (
                WeightedMSELoss() if self.readout_spec.mask_key is not None else MSELoss()
            )
        elif self.readout_spec.data_type == DataType.EVENT_RATE:
            self.loss_fn = PoissonNLLLoss(log_input=True, full=True, eps=1e-9, reduction="mean")
        else:
            raise ValueError(f"Unsupported data type: {self.readout_spec.data_type}")

    # ------------------------------------------------------------
    # Data
    # ------------------------------------------------------------

    def _drop_clipped_trials(self, intervals: dict, split: str, window: float) -> dict:
        """Drop trials narrower than ``window``.

        ``&`` truncates a trial straddling a split boundary rather than dropping it.
        One such trial is fatal: the model needs a fixed ``T``, and its batch dies in
        ``collate`` before :meth:`forward` can reach its own shape check.
        """
        kept, n_dropped = {}, 0
        for rid, iv in intervals.items():
            if len(iv) == 0:
                kept[rid] = iv
                continue
            # Tolerance absorbs float noise; a clipped trial misses by many bins.
            full = np.abs((iv.end - iv.start) - window) < 1e-6
            if full.all():
                kept[rid] = iv
                continue
            n_dropped += int((~full).sum())
            kept[rid] = iv.select_by_mask(full)

        if n_dropped:
            self.logger.warn(
                f"dropped {n_dropped} {split} trial(s) clipped to less than the "
                f"{window}s context window by a split boundary"
            )
        return kept

    def _setup_pooled_loader(self, split: Literal["train", "val"]):
        """Build a session-homogeneous loader over the pretrain sessions."""
        shuffle = split == "train"

        dataset = IBLBrainWideBenchSingleTaskBehavior(
            root=self.cfg.data_root,
            split=split,
            task=self.cfg.task,
            recording_ids=self.cfg.recording_ids,  # null uses every pretrain session
        )
        # True for both splits, so pretrain and finetune see the same distribution.
        intervals = self._drop_clipped_trials(
            dataset.get_trial_intervals(within_task_window=True), split, dataset.context_window
        )
        # A session with no target for this task is absent from intervals, not empty.
        dropped = sorted(set(dataset.recording_ids) - set(intervals))
        dropped += sorted(sid for sid, iv in intervals.items() if len(iv) == 0)
        if dropped:
            self.logger.warn(
                f"{len(dropped)} of {len(dataset.recording_ids)} session(s) contribute no "
                f"{split} trial for task {self.cfg.task}: {dropped[:5]}"
                f"{' ...' if len(dropped) > 5 else ''}"
            )

        batch_sampler = DistributedSamplerWrapper(
            SessionBatchSampler(
                TrialSampler(
                    sampling_intervals=intervals,
                    shuffle=shuffle,
                    generator=torch.Generator().manual_seed(self.cfg.seed),
                ),
                batch_size=max(self.cfg.batch_size // self.world_size, 1),
                shuffle_batches=shuffle,
                generator=torch.Generator().manual_seed(self.cfg.seed),
                drop_last=shuffle,
            )
        )
        has_workers = self.cfg.num_workers > 0
        loader = DataLoader(
            dataset,
            batch_sampler=batch_sampler,
            collate_fn=supervised_collate,
            num_workers=self.cfg.num_workers,
            pin_memory=self.cfg.pin_memory,
            prefetch_factor=self.cfg.prefetch_factor if has_workers else None,
            persistent_workers=self.cfg.persistent_workers if has_workers else False,
            multiprocessing_context=mp.get_context("fork") if has_workers else None,
        )

        setattr(self, f"{split}_dataset", dataset)
        setattr(self, f"{split}_loader", loader)
        self.logger.info(
            f"{split.capitalize()} dataset: {len(dataset.recording_ids) - len(dropped)} "
            f"contributing of {len(dataset.get_session_ids())} sessions, "
            f"{len(dataset.get_unit_ids())} units, {len(batch_sampler)} batches"
        )

    # ------------------------------------------------------------
    # Model
    # ------------------------------------------------------------

    def setup_model(self, ckpt: dict | None):
        self.model = hydra.utils.instantiate(self.cfg.model)
        if not isinstance(self.model, RRRDecoder):
            raise TypeError(
                f"{type(self).__name__} requires a RRRDecoder model, got {type(self.model).__name__}"
            )
        self.add_checkpoint_items(model=self.model)

        self.logger.info(f"Precision: {self.precision}")
        self.logger.info(f"Model: {self.model.__class__}")
        self.model.train()

    def link_model(self, model):
        for split in ["train", "val"]:
            model_transforms = hydra.utils.instantiate(self.cfg.get(f"{split}_transforms", []))
            dataset = getattr(self, f"{split}_loader").dataset
            if dataset.transform is None:
                dataset.transform = Compose([*model_transforms, model.input_fn])
            else:
                dataset.transform = Compose([dataset.transform, *model_transforms, model.input_fn])

        model.link_datasets(self.train_loader.dataset, self.val_loader.dataset, None)
        model.configure_readout(self.readout_spec)

        num_params, _ = get_num_params(self.model)
        shared = self.model.V.numel()
        self.logger.info(
            f"Number of parameters: {human_readable(num_params)} ({num_params:,}), of "
            f"which shared V: {shared:,} ({100 * shared / num_params:.1f}%); the rest "
            f"is per-session U/b and does not transfer"
        )

    def setup_optimizers(self, ckpt: dict | None):
        grouped_parameters = self.get_param_groups()

        self.optimizer = torch.optim.AdamW(grouped_parameters, lr=self.cfg.base_lr)
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
            mask = y["mask"].unsqueeze(-1).expand(-1, -1, self.readout_spec.dim)
            return self.loss_fn(pred, target, mask.reshape(mask.shape[0], -1))
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
                loss = self.loss(self.model(**X["model_inputs"]), X, y)
            loss.backward()
            self.clip_and_log_grad_norm()
            self.optimizer.step()
            self.scheduler.step()
            self.train_step += 1

            epoch_losses.append(loss.item())
            if self.cfg.log_train_step:
                self.logger.log("train/loss", epoch_losses[-1], pbar=True)
                self.logger.log("train/lr", self.scheduler.get_last_lr()[0], pbar=False)
                self.push_logs()

        if not self.cfg.log_train_step:
            self.logger.log("train/loss", sum(epoch_losses) / max(len(epoch_losses), 1))

    @torch.inference_mode()
    def val_epoch(self):
        """Score pooled over sessions; ``best.pt`` is what the probe loads."""
        self.model.eval()
        val_metrics = self._eval_epoch(self.val_loader, split="val")
        primary = val_metrics[self.readout_spec.primary_metric]

        improved = self.save_if_best(primary, metrics=val_metrics)
        early_stop = self.step_patience(improved)

        self.logger.log_dict(
            {f"val/{k}": v for k, v in val_metrics.items()}
            | {"val/avg": primary, "val/epoch": self.epoch}
        )
        self.push_logs()
        self.return_value = val_metrics
        return early_stop

    def _eval_epoch(self, loader: DataLoader, split: Literal["val"] = "val"):
        metrics = {
            name: metric().to(self.device) for name, metric in self.readout_spec.metrics.items()
        }

        for X, y in self.logger.get_pbar(loader, prefix=split):
            X = move_to_device(X, device=self.device)
            y = move_to_device(y, device=self.device)
            with torch.autocast(device_type="cuda", dtype=self.precision.dtype):
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
                    metric.update(pred[mask], target[mask])
                elif self.readout_spec.data_type in (DataType.BINARY, DataType.MULTINOMIAL):
                    # K-way classification needs (B,) label indices
                    metric.update(pred, target.squeeze(-1))
                else:
                    metric.update(pred, target)

        return aggregate_metrics(metrics)

    def log_training_results(self):
        self.logger.info(
            f"Best val {self.readout_spec.primary_metric} {self.best_val:.5f} at epoch "
            f"{self.best_epoch}. The checkpoint's shared basis V can be transferred with "
            f"trainer=rrr_probe ckpt.load_from=<run_id>/best.pt"
        )
