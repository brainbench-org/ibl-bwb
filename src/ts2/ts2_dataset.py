from typing import Any, get_args

import numpy as np
from torch_brain.data import Data
from torch_brain.utils.binning import bin_spikes

from core.dataset import IBLBrainWideBench2026, Split, VariableContextMixin
from ibl_bwb_eval.tasks import TS2Task
from ibl_bwb_eval.tasks.ts2 import COSMOOTH_TARGET_WINDOW, FORECAST_BASE_WINDOW


class IBLBrainWideBenchTS2(VariableContextMixin, IBLBrainWideBench2026):
    """Dataset for the IBL BrainWideBench benchmark.

    Args:
        root: The root directory of the dataset.
        split: The split of the dataset (train, val, test).
        dirname: The name of the dataset (and the directory containing its data).
        recording_ids: The recording ids to include in the dataset. If None, all
                       recordings in the dataset related to the regime are included.
        transform: The transform(s) to apply to the data.
        task: The task to include in the dataset.
        mask_token: Placeholder value for masked entries, for models that need one.
        mask_input: Whether the held-out spikes are removed from the model input.
                    Only honored on ``val``: False leaks the held-out activity into
                    the input, for models that cannot take a corrupted one (AEs).
        context_length: Total sampled window size in seconds (preceding context plus the
                        fixed 1s target). Defaults to 1.0 (no extra context).
    """

    BIN_SIZE = 0.02
    FORECAST_RATIO = 0.1
    CO_SMOOTHING_HELD_OUT_RATIO = 0.1
    SPLITS: tuple[Split | None, ...] = ("train", "val", "test")

    def __init__(
        self,
        root: str,
        split: Split,
        recording_id: str,
        dirname: str = "ibl_brain_wide_bench_2026",
        transform: Any | None = None,
        task: TS2Task | None = None,
        mask_token: float = float("nan"),
        mask_input: bool = True,
        context_length: float = 1.0,
    ):
        assert recording_id is not None and isinstance(recording_id, str)

        assert task is not None, "No Task"
        assert task in get_args(TS2Task), f"{task} not in TS2Task"
        self.task = task
        # co_smoothing's TARGET_WINDOW is itself the scored window; forecasting's is
        # a fraction of the base window (which also defines the minimum context length).
        # This is set before super().__init__() for the mixin's context_length bound.
        self.TARGET_WINDOW = (
            COSMOOTH_TARGET_WINDOW if task == "co_smoothing" else FORECAST_BASE_WINDOW
        )

        super().__init__(
            root=root,
            dirname=dirname,
            recording_ids=recording_id,
            transform=transform,
            split=split,
            regime="eval",
            require_unit_filtering="selected_units",
            contract="TS2",
            context_length=context_length,
        )

        self.recording_id = recording_id

        self.return_target = True

        self.mask_token = mask_token
        # the test contract is fixed, only val may opt out
        self.mask_input = mask_input if split == "val" else True

    def dataset_transform(self, data: Data) -> Data:
        keep = f"ts2_co_smoothing_is_held_out_{self.split}" if self.task == "co_smoothing" else None

        for attr in ("ts2_co_smoothing_is_held_out_val", "ts2_co_smoothing_is_held_out_test"):
            if attr != keep and hasattr(data.units, attr):
                delattr(data.units, attr)

        return data

    def _get_target(self, data: Data) -> dict:
        spikes = data.spikes

        bin_counts = bin_spikes(
            spikes,
            num_units=len(data.units),
            bin_size=self.BIN_SIZE,
            dtype=np.float32,
        )

        target = {"values": bin_counts}

        if self.split == "train":
            return target

        T, N = bin_counts.shape

        if self.task == "co_smoothing":
            # The scored region is the trailing COSMOOTH_TARGET_WINDOW.
            target_bins = round(COSMOOTH_TARGET_WINDOW / self.BIN_SIZE)

            attr = f"ts2_co_smoothing_is_held_out_{self.split}"
            assert hasattr(data.units, attr), f"{attr} is not in the data"
            held_out_units = getattr(data.units, attr)

            # False everywhere before the trailing target_bins
            mask_bin_counts = np.zeros((T, N), dtype=bool)
            mask_bin_counts[-target_bins:] = held_out_units  # (N,) -> (target_bins, N)

            mask_spikes = ~np.isin(spikes.unit_index, np.where(held_out_units)[0])

            target["mask_units"] = held_out_units

        elif self.task == "forecasting":
            # The scored region is a fixed FORECAST_RATIO of FORECAST_BASE_WINDOW
            forecast_bins = round(self.FORECAST_RATIO * FORECAST_BASE_WINDOW / self.BIN_SIZE)
            mask_bin_counts = np.zeros((T, N), dtype=bool)
            mask_bin_counts[-forecast_bins:] = True

            end = spikes.domain.end[-1]
            mask_timestamps = end - self.FORECAST_RATIO * FORECAST_BASE_WINDOW
            mask_spikes = spikes.timestamps < mask_timestamps

            target["mask_timestamps"] = mask_timestamps

        target["window_start"] = data.absolute_start
        target["mask"] = mask_bin_counts

        if self.mask_input:
            data.spikes = spikes.select_by_mask(mask_spikes)

        return target

    def get_sampling_intervals(self):
        """The intervals a sampler may draw windows from: this split's task-aligned trials."""
        recording = self.get_recording(self.recording_id)

        intervals = recording.task_aligned_intervals.domain
        intervals = intervals & getattr(recording, f"ts2_{self.split}_domain")

        return {self.recording_id: intervals}
