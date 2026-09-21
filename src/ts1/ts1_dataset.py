import logging
from typing import Any, get_args

import numpy as np
from torch_brain.data import Data

from core.dataset import IBLBrainWideBench2026, Split, VariableContextMixin
from ibl_bwb_eval.tasks import (
    TARGET_WINDOW as TS1_TARGET_WINDOW,
)
from ibl_bwb_eval.tasks import (
    TargetResolution,
    TS1Task,
    get_ts1_readout_spec,
)


class IBLBrainWideBenchTS1(VariableContextMixin, IBLBrainWideBench2026):
    """Dataset for the IBL BrainWideBench benchmark.

    Args:
        root: The root directory of the dataset.
        split: The split of the dataset (train, val, test).
        task: The task to include in the dataset. One task at a time.
        recording_id: The recording id to include in the dataset.
        dirname: The name of the dataset (and the directory containing its data).
        transform: The transform(s) to apply to the data.
        normalize_behavior: Whether to normalize the behavior data. This should be set to True
                            unless the user wants to use custom target preprocessing.
        context_length: Total sampled window size in seconds (preceding context plus the
                        fixed 1s target). Defaults to 1.0 (no extra context).
    """

    SPLITS: tuple[Split | None, ...] = ("train", "val", "test")
    TARGET_WINDOW = TS1_TARGET_WINDOW

    def __init__(
        self,
        root: str,
        split: Split,
        task: TS1Task,
        recording_id: str,
        dirname: str = "ibl_brain_wide_bench_2026",
        transform: Any | None = None,
        normalize_behavior: bool = True,
        context_length: float = 1.0,
    ):
        assert recording_id is not None and isinstance(recording_id, str), (
            "Only one recording_id is allowed at a time for TS1 eval"
        )

        super().__init__(
            root=root,
            dirname=dirname,
            recording_ids=recording_id,
            transform=transform,
            split=split,
            regime="eval",
            contract="TS1",
            context_length=context_length,
        )
        self.return_target = True

        # TS1 decodes from the full population, but any build is allowed
        if self.unit_filtering.label != "all_units":
            logging.warning(
                f"TS1 baselines are all reported on the all_units build, got "
                f"{self.unit_filtering.describe()}. Report the build with your results so "
                "the difference in setting is clear."
            )

        assert task in self.get_ts1_supported_tasks(), (
            f"Error: Task {task} is not supported by the benchmark!"
        )

        self.task = task
        self.recording_id = recording_id
        self.normalize_behavior = normalize_behavior

    @classmethod
    def get_ts1_supported_tasks(
        cls, target_resolution: TargetResolution | None = None
    ) -> list[TS1Task]:
        """Returns the tasks the benchmark supports, as the string values of TS1Task.

        Args:
            target_resolution: If given, keep only the tasks with this target resolution.

        Returns:
            list[TS1Task]: The supported tasks.

        Example:
            >>> tasks = IBLBrainWideBenchTS1.get_ts1_supported_tasks()
            >>> print(tasks)
            ['choice', ...]
        """
        if target_resolution is None:
            return list(get_args(TS1Task))

        assert target_resolution in TargetResolution, (
            f"{target_resolution} is not a valid TargetResolution"
        )

        filtered_tasks = [
            task
            for task in get_args(TS1Task)
            if get_ts1_readout_spec(task).target_resolution == target_resolution
        ]

        return filtered_tasks

    def dataset_transform(self, data: Data) -> Data:
        """Defines dataset-level transformations that are applied to all recordings in the Dataset.

        This method can be applied on an entire recording or a single slice.

        Args:
            data: The Data object to apply the transformations to.

        Returns:
            The Data object with the transformations applied.

        Example:
            >>> dataset = IBLBrainWideBenchTS1(...)
            >>> data = dataset.get_recording(...)
            >>> data = dataset.dataset_transform(data)
            >>> slice = data.slice(0.5, 1.5)
            >>> slice = dataset.dataset_transform(slice)
        """
        readout_spec = get_ts1_readout_spec(self.task)
        assert data.has_nested_attribute(readout_spec.value_key)

        target = data.get_nested_attribute(readout_spec.value_key)

        if self.normalize_behavior:
            if self.task == "licking_rate":
                target = np.round(target / 50.0).astype(np.int32)
            elif (
                readout_spec.target_resolution == TargetResolution.TIMESTEP
                and self.task != "wheel_speed"
            ):
                normalization = data.get_nested_attribute(f"ts1_normalize.{readout_spec.value_key}")
                target = (target - normalization.mean) / normalization.std

        if target.dtype == np.float64:
            target = target.astype(np.float32)

        data.set_nested_attribute(readout_spec.value_key, target)

        return data

    def _get_target(self, data: Data) -> dict:
        """Extract the target(s) from the data based on the readout specification.

        Args:
            data: The Data object to extract the target(s) from.

        Returns:
            A dictionary containing the target(s).
            The dictionary contains the following keys:
            - "values": The value of the target(s).
            - "timestamps": The timestamps of the target(s), when the readout spec defines them.
            - "mask": The mask of the target(s), when the readout spec defines one.
            - "trial_id": The trial index of the target(s), when the data carries one.
        """
        readout_spec = get_ts1_readout_spec(self.task)

        # The target is the trailing TARGET_WINDOW of whatever data we got, which may
        # be shorter than context_window if __getitem__ clipped it.
        data_end = data.domain.end[-1]
        context_margin = data_end - TS1_TARGET_WINDOW
        target_data = data.slice(context_margin, data_end) if context_margin > 1e-6 else data

        target = {
            "values": target_data.get_nested_attribute(readout_spec.value_key),
        }
        if readout_spec.timestamp_key is not None:
            target["timestamps"] = target_data.get_nested_attribute(readout_spec.timestamp_key)
        if readout_spec.mask_key is not None:
            target["mask"] = target_data.get_nested_attribute(readout_spec.mask_key)
        trial_id_key = f"{readout_spec.interval_key}.trial_index"
        if target_data.has_nested_attribute(trial_id_key):
            target["trial_id"] = target_data.get_nested_attribute(trial_id_key)

        if self.split == "test":
            data.delete_nested_attribute(readout_spec.value_key)

        return target

    def get_sampling_intervals(self):
        """The intervals a sampler may draw windows from: this split's task-aligned trials."""
        recording = self.get_recording(self.recording_id)

        intervals = recording.task_aligned_intervals.domain
        intervals = intervals & getattr(recording, f"ts1_{self.split}_domain")

        readout_spec = get_ts1_readout_spec(self.task)

        assert recording.has_nested_attribute(readout_spec.value_key)

        task_window = recording.get_nested_attribute(readout_spec.interval_key)

        if readout_spec.mask_counts_key is not None:
            counts = recording.get_nested_attribute(readout_spec.mask_counts_key)
            task_window = task_window.select_by_mask(counts > 0)

        intervals = intervals & task_window

        task_domain = recording.get_nested_attribute(readout_spec.domain_key)
        intervals = intervals & task_domain

        return {self.recording_id: intervals}
