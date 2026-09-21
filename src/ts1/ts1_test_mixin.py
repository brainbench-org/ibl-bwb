import multiprocessing as mp
from pathlib import Path
from typing import final

import torch
from omegaconf import OmegaConf
from torch.utils.data import DataLoader
from torch_brain.samplers import TrialSampler

from core.batching.collate import supervised_collate
from core.samplers import DistributedSamplerWrapper
from core.utils.util import move_to_device
from ibl_bwb_eval.metrics import aggregate_metrics
from ibl_bwb_eval.predictions import PredictionsWriter
from ibl_bwb_eval.tasks import DataType, TargetResolution, task_id
from ts1.ts1_dataset import IBLBrainWideBenchTS1


class TS1TestMixin:
    """Mixin enforcing the TS1 benchmark test contract.

    Declares setup_test_loader and test as final so no subclass
    can alter what data is evaluated or how results are computed.
    Validation strategy is intentionally left to the concrete trainer.

    Assumes the subclass also inherits from core.trainer.BaseTrainer (for self.device,
    self.logger, self.precision, self.push_logs, self.rank, etc.) and that the
    following attributes are initialized before calling test:

    - self.task
    - self.readout_spec
    - self.model
    - self.best_model
    - self.test_dataset
    - self.test_loader
    - self.best_metrics
    - self.return_value
    - self.cfg

    and the following method is implemented:

    - self.predict(X, target_timestamps=None, target_mask=None)
    """

    @final
    def setup_test_loader(self):
        self.test_dataset = IBLBrainWideBenchTS1(
            root=self.cfg.data_root,
            split="test",
            recording_id=self.cfg.recording_id,
            task=self.task,
            context_length=OmegaConf.select(self.cfg, "context_length", default=1.0),
        )
        self.test_sampler = DistributedSamplerWrapper(
            TrialSampler(
                sampling_intervals=self.test_dataset.get_sampling_intervals(),
                shuffle=False,
            )
        )
        has_workers = self.cfg.num_workers > 0
        self.test_loader = DataLoader(
            self.test_dataset,
            sampler=self.test_sampler,
            collate_fn=supervised_collate,
            batch_size=self.cfg.batch_size // self.world_size,
            num_workers=self.cfg.num_workers,
            pin_memory=self.cfg.pin_memory,
            persistent_workers=self.cfg.persistent_workers if has_workers else False,
            multiprocessing_context=mp.get_context("fork") if has_workers else None,
        )
        self.logger.info(
            f"Test dataset: {len(self.test_dataset.get_session_ids())} sessions, {len(self.test_dataset.get_unit_ids())} units"
        )
        self.logger.info(
            f"Testing on {len(self.test_loader)} batches, total {len(self.test_sampler)} samples"
        )

    @final
    @torch.inference_mode()
    def test(self):
        # load best model
        if self.best_model is not None:
            self.model.load_state_dict(self.best_model)

        self.model.eval()
        with torch.no_grad():
            metrics = {
                name: metric().to(self.device) for name, metric in self.readout_spec.metrics.items()
            }

            preds = PredictionsWriter(
                enable=OmegaConf.select(self.cfg, "save_preds.enable", default=False),
                base_path=Path(OmegaConf.select(self.cfg, "save_preds.path", default="")),
                task=task_id("ts1", self.task),
                seed=self.cfg.seed,
                rank=self.rank,
                label=OmegaConf.select(self.cfg, "save_preds.label", default=None),
                metadata={
                    "recording_id": self.cfg.recording_id,
                    "unit_filtering": self.test_dataset.unit_filtering.label,
                    "dataset_version": self.test_dataset.dataset_version,
                },
                path_fn=lambda p, meta: p / meta["recording_id"],
            )

            for X, y in self.logger.get_pbar(self.test_loader, prefix="test"):
                X = move_to_device(X, device=self.device)
                y = move_to_device(y, device=self.device)
                with torch.autocast(device_type="cuda", dtype=self.precision.dtype):
                    pred = self.predict(X, y.get("timestamps"), y.get("mask"))
                    target = y["values"]

                    # capture pred in (B, T, D) / (B, 1, D) form before metric reshape
                    pred_for_save = pred.cpu().float()

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

                # pred_for_save: (B, T, D) timestep-level  |  (B, 1, D) sequence-level
                # trial_id:      (B, 1) -> (B,)
                # timestamps:    (B, T) or (B, 1) when present
                preds.add(predictions=pred_for_save, trial_id=y["trial_id"].squeeze(-1))
                if "timestamps" in y:
                    preds.add(timestamps=y["timestamps"])

        preds.save(logger=self.logger)

        test_metrics = aggregate_metrics(metrics)
        self.best_test_metrics = {f"best/test/{k}": v for k, v in test_metrics.items()}
        self.best_test_metrics["best/test/avg"] = test_metrics[self.readout_spec.primary_metric]
        self.best_metrics |= self.best_test_metrics
        self.return_value |= self.best_test_metrics
        self.logger.log_dict(self.best_metrics)
        self.push_logs()
