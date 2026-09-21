import multiprocessing as mp
from pathlib import Path
from typing import final

import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader
from torch_brain.batching import collate
from torch_brain.samplers import SequentialFixedWindowSampler

from core.model import BaseModel
from core.utils.logger import Logger
from core.utils.util import Precision, move_to_device
from ibl_bwb_eval.entity_ids import encode_entity_ids
from ibl_bwb_eval.metrics import aggregate_metrics
from ibl_bwb_eval.predictions import PredictionsWriter
from ibl_bwb_eval.tasks import TS2Task, get_ts2_readout_spec, task_id
from ts2.ts2_dataset import IBLBrainWideBenchTS2


def select_scored_entries(
    pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor, mask_dim: int
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Keep only the held-out entries of a ``(batch, time, units)`` pair.

    ``mask`` is constant along every dim but ``mask_dim``, so one row of it names the
    held-out units (co-smoothing) or timesteps (forecasting). Returns the flattened
    pred/target the metrics score, the pred still in window form for the submission,
    and the kept indices.

    Uses the last timestep as the representative row: co-smoothing's mask is only True
    in the trailing TARGET_WINDOW, False everywhere before it once context_window adds
    preceding context.
    """
    keep = mask.movedim(mask_dim, -1)[0, -1].nonzero(as_tuple=True)[0]
    pred_subset = pred.index_select(mask_dim, keep)
    target_subset = target.index_select(mask_dim, keep)
    return (
        pred_subset.reshape(-1, pred_subset.shape[-1]),
        target_subset.reshape(-1, target_subset.shape[-1]),
        pred_subset,
        keep,
    )


class TS2TestMixin:
    """Mixin enforcing the TS2 benchmark test contract.

    Declares setup_test_loader and test as final so no subclass
    can alter what data is evaluated or how results are computed.
    Validation strategy is intentionally left to the concrete trainer.

    Assumes the subclass also inherits from core.trainer.BaseTrainer (for self.device,
    self.logger, self.precision, self.push_logs, self.rank, etc.) and that the
    following attributes are initialized before calling test:

    - self.task
    - self.model
    - self.best_model
    - self.test_loader
    - self.best_metrics
    - self.return_value
    - self.cfg

    and the following method is implemented:

    - self.predict(X, mask=None, mask_timestamps=None, mask_units=None)
    """

    # from Trainer
    device: torch.device
    rank: int
    epoch: int
    logger: Logger
    precision: Precision

    # declared by concrete trainer subclasses
    task: TS2Task
    model: BaseModel
    best_model: dict | None
    test_loader: DataLoader
    cfg: DictConfig
    best_metrics: dict
    return_value: dict

    def predict(
        self,
        _X: dict,
        _mask: torch.Tensor | None = None,
        _mask_timestamps: torch.Tensor | None = None,
        _mask_units: torch.Tensor | None = None,
    ) -> torch.Tensor:
        raise NotImplementedError

    @final
    def setup_test_loader(self):
        self.test_dataset = IBLBrainWideBenchTS2(
            root=self.cfg.data_root,
            split="test",
            recording_id=self.cfg.recording_id,
            task=self.task,
            context_length=OmegaConf.select(self.cfg, "context_length", default=1.0),
        )
        # One window per bin: the scorer aligns to the ground truth by window start,
        # so a coarser step scores a subset of it.
        self.test_sampler = SequentialFixedWindowSampler(
            sampling_intervals=self.test_dataset.get_sampling_intervals(),
            window_length=self.test_dataset.TARGET_WINDOW,
            step=self.test_dataset.BIN_SIZE,
        )
        has_workers = self.cfg.num_workers > 0
        self.test_loader = DataLoader(
            self.test_dataset,
            sampler=self.test_sampler,
            collate_fn=collate,
            batch_size=self.cfg.batch_size,
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
        if self.best_model is not None:
            self.model.load_state_dict(self.best_model)
        self.model.eval()
        spec = get_ts2_readout_spec(self.task)
        metrics = {name: metric().to(self.device) for name, metric in spec.metrics.items()}

        preds = PredictionsWriter(
            enable=OmegaConf.select(self.cfg, "save_preds.enable", default=False),
            base_path=Path(OmegaConf.select(self.cfg, "save_preds.path", default="")),
            task=task_id("ts2", self.task),
            seed=self.cfg.seed,
            rank=self.rank,
            label=OmegaConf.select(self.cfg, "save_preds.label", default=None),
            metadata={
                "recording_id": self.cfg.recording_id,
                "unit_filtering": self.test_dataset.unit_filtering.label,
                "dataset_version": self.test_dataset.dataset_version,
                "context_window": self.test_dataset.context_window,
                "bin_size": self.test_dataset.BIN_SIZE,
            },
            path_fn=lambda p, meta: p / meta["recording_id"],
        )

        keep = None

        for X, y in self.logger.get_pbar(self.test_loader, prefix="test"):
            X = move_to_device(X, device=self.device)
            y = move_to_device(y, device=self.device)

            with torch.autocast(device_type=self.device.type, dtype=self.precision.dtype):
                target = y["values"]
                mask = y["mask"]

                if mask.sum() == 0:
                    continue

                pred = self.predict(X, mask, y.get("mask_timestamps"), y.get("mask_units"))

                assert mask.dtype == torch.bool, "Mask must be a boolean tensor"

                pred, target, pred_subset, keep = select_scored_entries(
                    pred, target, mask, spec.mask_dim
                )

                for metric in metrics.values():
                    metric.update(pred, target)

                preds.add(predictions=pred_subset, dtype=torch.float16)
                # Note: it's important to retain float64 precision here
                preds.add(window_timestamps=y["window_start"].cpu(), dtype=torch.float64)  # (B,)

        if keep is not None:
            rec = self.test_dataset.get_recording(self.cfg.recording_id)
            all_unit_ids = np.array(rec.units.id)
            # co-smoothing scores a subset of the units; forecasting scores all of them,
            # over a subset of the timesteps
            scored = all_unit_ids[keep.cpu().numpy()] if spec.mask_dim == 2 else all_unit_ids
            preds.set(unit_ids=encode_entity_ids(scored))

        preds.save(logger=self.logger)

        test_metrics = aggregate_metrics(metrics)

        self.best_test_metrics = {f"best/test/{k}": v for k, v in test_metrics.items()}
        self.best_test_metrics["best/test/avg"] = test_metrics[spec.primary_metric]
        self.best_metrics |= self.best_test_metrics
        self.return_value |= self.best_test_metrics
        self.logger.log_dict(self.best_metrics)
        self.push_logs()
