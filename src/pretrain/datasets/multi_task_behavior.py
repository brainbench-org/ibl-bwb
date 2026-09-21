from functools import reduce
from operator import or_
from typing import Any

import numpy as np
from omegaconf import ListConfig
from torch_brain.data import Data, Interval

from core.dataset import IBLBrainWideBench2026, Split
from ibl_bwb_eval.tasks import (
    BEHAVIOR_SFREQ,
    TargetResolution,
    TS1Task,
    get_ts1_readout_spec,
    get_ts1_supported_tasks,
)


class IBLBrainWideBenchMultiTaskBehavior(IBLBrainWideBench2026):
    """Spikes aligned to several behavioral targets at once.

    ``__getitem__`` returns a single dict rather than an ``(input, target)`` pair,
    since the model's ``input_fn`` emits the targets. Two samplings are offered:
    :meth:`get_task_align_intervals` stacks one window per (task, trial), and
    :meth:`get_trial_intervals` merges the regions where any task has data.

    Args:
        root: Directory holding the build.
        split: Which pretrain split to read, ``train`` or ``val``.
        dirname: Name of the build directory under ``root``.
        recording_ids: Recordings to load, or None for every pretrain recording.
        transform: Applied to each item, the model's ``input_fn`` among them.
        tasks: The TS1 tasks to align to, one or a list of them.
        normalize_behavior: Whether to normalize the targets. Timestep-level signals are
            z-scored against the recording's pretrain statistics, ``wheel_speed``
            excepted; ``licking_rate`` is rounded into counts instead.
    """

    SPLITS: tuple[Split | None, ...] = ("train", "val")

    def __init__(
        self,
        root: str,
        split: Split,
        tasks: TS1Task | list[TS1Task],
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
        # For now targets are emitted by the model's ``input_fn`` (as ``target_values``
        # / ``target_masks``), so ``__getitem__`` returns a single dict rather than an
        # ``(input, target)`` pair. A caller wanting the pair flips this and implements
        # ``_get_target``.
        self.return_target = False

        if isinstance(tasks, ListConfig):
            tasks = list(tasks)

        if not isinstance(tasks, list):
            tasks = [tasks]

        for task in tasks:
            assert task in get_ts1_supported_tasks(), (
                f"Error: Task {task} is not supported by the benchmark!"
            )

        self.tasks = tasks
        self.normalize_behavior = normalize_behavior

    def dataset_transform(self, data: Data) -> Data:
        if not self.normalize_behavior:
            return data

        for task in self.tasks:
            readout_spec = get_ts1_readout_spec(task)

            if not data.has_nested_attribute(readout_spec.value_key):
                continue

            target = data.get_nested_attribute(readout_spec.value_key)

            if task == "licking_rate":
                target = np.round(target / BEHAVIOR_SFREQ).astype(np.int32)
            elif (
                readout_spec.target_resolution == TargetResolution.TIMESTEP
                and task != "wheel_speed"
            ):
                normalization = data.get_nested_attribute(
                    f"pretrain_normalize.{readout_spec.value_key}"
                )
                target = (target - normalization.mean) / normalization.std

            if target.dtype == np.float64:
                target = target.astype(np.float32)

            data.set_nested_attribute(readout_spec.value_key, target)

        return data

    def get_task_align_intervals(self):
        sampling_intervals = {}
        for recording_id in self.recording_ids:
            recording = self.get_recording(recording_id)
            split_intervals = getattr(recording, f"pretrain_{self.split}_domain")

            intervals = []
            for task in self.tasks:
                readout_spec = get_ts1_readout_spec(task)
                if not recording.has_nested_attribute(
                    readout_spec.interval_key
                ) or not recording.has_nested_attribute(readout_spec.domain_key):
                    continue

                interval = recording.get_nested_attribute(readout_spec.interval_key)
                target_domain = recording.get_nested_attribute(readout_spec.domain_key)

                interval = interval & target_domain & split_intervals

                intervals.append(interval)

            # Stack rather than merge: one window per (task, trial), so overlapping
            # tasks each keep their own. Intentionally overlapping and unsorted.
            flattened = [(start, end) for interval in intervals for start, end in interval]

            if len(flattened) != 0:
                sampling_intervals[recording_id] = Interval.from_list(flattened)
            else:
                sampling_intervals[recording_id] = Interval(
                    start=np.array([]),
                    end=np.array([]),
                )

        return sampling_intervals

    def get_trial_intervals(self):
        """Merged regions to sample fixed-length windows from.

        Restricted to where at least one task has data. For
        :class:`RandomFixedWindowSampler`, unlike the stacked per-trial windows of
        :meth:`get_task_align_intervals`.
        """
        sampling_intervals = {}
        for recording_id in self.recording_ids:
            recording = self.get_recording(recording_id)

            intervals = recording.task_aligned_intervals.domain
            intervals = intervals & getattr(recording, f"pretrain_{self.split}_domain")

            task_domains = [
                recording.get_nested_attribute(get_ts1_readout_spec(task).domain_key)
                for task in self.tasks
                if recording.has_nested_attribute(get_ts1_readout_spec(task).domain_key)
            ]

            if len(task_domains) == 0:
                sampling_intervals[recording_id] = Interval(
                    start=np.array([]),
                    end=np.array([]),
                )
                continue

            sampling_intervals[recording_id] = intervals & reduce(or_, task_domains)

        return sampling_intervals
