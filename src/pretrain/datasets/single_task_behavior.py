from typing import Any

import numpy as np
from torch_brain.data import Data

from core.dataset import IBLBrainWideBench2026, Split
from ibl_bwb_eval.tasks import (
    BEHAVIOR_SFREQ,
    TargetResolution,
    TS1Task,
    get_ts1_readout_spec,
    get_ts1_supported_tasks,
)


class IBLBrainWideBenchSingleTaskBehavior(IBLBrainWideBench2026):
    """Spikes aligned to one behavioral target.

    ``__getitem__`` returns an ``(input, target)`` pair, the target being ``task``.

    Args:
        root: Directory holding the build.
        split: Which pretrain split to read, ``train`` or ``val``.
        dirname: Name of the build directory under ``root``.
        recording_ids: Recordings to load, or None for every pretrain recording.
        transform: Applied to each item, the model's ``input_fn`` among them.
        task: The TS1 task whose behavioral signal is the target.
        normalize_behavior: Whether to normalize the target. Timestep-level signals are
            z-scored against the recording's pretrain statistics, ``wheel_speed``
            excepted; ``licking_rate`` is rounded into counts instead.
    """

    SPLITS: tuple[Split | None, ...] = ("train", "val")

    def __init__(
        self,
        root: str,
        split: Split,
        task: TS1Task,
        dirname: str = "ibl_brain_wide_bench_2026",
        recording_ids: str | list[str] | None = None,
        transform: Any | None = None,
        normalize_behavior: bool = True,
    ):
        super().__init__(
            root=root,
            dirname=dirname,
            recording_ids=recording_ids,
            transform=transform,
            split=split,
            regime="pretrain",
        )
        self.return_target = True

        assert task in get_ts1_supported_tasks(), (
            f"Error: Task {task} is not supported by the benchmark!"
        )

        self.task = task
        self.normalize_behavior = normalize_behavior

    def dataset_transform(self, data: Data) -> Data:
        if not self.normalize_behavior:
            return data

        readout_spec = get_ts1_readout_spec(self.task)

        if not data.has_nested_attribute(readout_spec.value_key):
            return data

        target = data.get_nested_attribute(readout_spec.value_key)

        if self.task == "licking_rate":
            target = np.round(target / BEHAVIOR_SFREQ).astype(np.int32)
        elif (
            readout_spec.target_resolution == TargetResolution.TIMESTEP
            and self.task != "wheel_speed"
        ):
            normalization = data.get_nested_attribute(
                f"pretrain_normalize.{readout_spec.value_key}"
            )
            target = (target - normalization.mean) / normalization.std

        if target.dtype == np.float64:
            target = target.astype(np.float32)

        data.set_nested_attribute(readout_spec.value_key, target)

        return data

    def _get_target(self, data: Data) -> dict:
        readout_spec = get_ts1_readout_spec(self.task)

        target = {
            "values": data.get_nested_attribute(readout_spec.value_key),
        }
        if readout_spec.timestamp_key is not None:
            target["timestamps"] = data.get_nested_attribute(readout_spec.timestamp_key)
        if readout_spec.mask_key is not None:
            target["mask"] = data.get_nested_attribute(readout_spec.mask_key)

        return target

    def get_trial_intervals(self, within_task_window: bool = True):
        sampling_intervals = {}
        for recording_id in self.recording_ids:
            recording = self.get_recording(recording_id)

            intervals = recording.task_aligned_intervals.domain
            intervals = intervals & getattr(recording, f"pretrain_{self.split}_domain")

            readout_spec = get_ts1_readout_spec(self.task)

            if not recording.has_nested_attribute(readout_spec.value_key):
                continue

            if within_task_window:
                task_window = recording.get_nested_attribute(readout_spec.interval_key)
                intervals = intervals & task_window

            task_domain = recording.get_nested_attribute(readout_spec.domain_key)
            sampling_intervals[recording_id] = intervals & task_domain

        return sampling_intervals
