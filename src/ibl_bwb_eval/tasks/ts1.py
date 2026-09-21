"""TS1: the task vocabulary, and what each task predicts.

Part of the evaluation contract rather than of any trainer, so the scorer, the TS1
suite and the multi-task pretraining datasets all read the same definition.
"""

from collections.abc import Callable
from dataclasses import dataclass
from functools import partial
from typing import Literal, TypeAlias, get_args

from ibl_bwb_eval.tasks.types import DataType, TargetResolution

TS1Task: TypeAlias = Literal[
    "choice",
    "reward",
    "stimulus_contrast",
    "whisker_motion_energy",
    "wheel_speed",
    "right_paw_speed",
    "left_paw_speed",
    "licking_rate",
]
"""The eight scored TS1 tasks: three sequence-level, five timestep-level."""

# The scored window and the rate its timestep-level targets are sampled at. Part of the
# evaluation contract, so they live here rather than on a dataset class.
BEHAVIOR_SFREQ = 50  # Hz
TARGET_WINDOW = 1.0  # seconds


@dataclass
class TS1ReadoutSpec:
    """Specification for a single decoding readout head.

    Defines the target variable's type, dimensionality, data keys, and evaluation
    metrics for one task in the benchmark.

    Attributes:
        id: Unique identifier for the readout (typically the task name).
        dim: Output dimensionality of the readout head.
        data_type: Type of the target variable (continuous, binary, etc.).
        target_resolution: Whether the target is at the item or timestep level.
        primary_metric: Name of the metric used for model selection and early stopping.
        metrics: Mapping from metric name to a callable that instantiates the metric.
        value_key: Key in the data dict for the target values.
        domain_key: Key in the data dict for the domain identifier.
        interval_key: Key in the data dict for the trial intervals.
        timestamp_key: Key in the data dict for target timestamps. Optional.
        mask_key: Key in the data dict for the validity mask. Optional.
        mask_counts_key: Key in the data dict for the mask counts. Optional.
    """

    id: str
    dim: int
    data_type: DataType
    target_resolution: TargetResolution
    # metrics
    primary_metric: str
    metrics: dict[str, Callable]
    # data keys
    value_key: str
    domain_key: str
    interval_key: str
    timestamp_key: str | None = None
    mask_key: str | None = None
    mask_counts_key: str | None = None

    @property
    def num_timesteps(self) -> int:
        """Target steps the readout emits. Derived, so it cannot drift from the resolution."""
        if self.target_resolution == TargetResolution.SEQUENCE:
            return 1
        return int(BEHAVIOR_SFREQ * TARGET_WINDOW)


def get_ts1_readout_spec(task: TS1Task) -> TS1ReadoutSpec:
    """What ``task`` predicts, and the metrics it is scored on."""
    from torchmetrics import (
        Accuracy,
        AveragePrecision,
        F1Score,
        MeanAbsoluteError,
        PearsonCorrCoef,
        R2Score,
    )

    from ibl_bwb_eval.metrics import PoissonD2Score

    match task:
        # classification tasks
        case "choice":
            return TS1ReadoutSpec(
                id="choice",
                dim=2,
                data_type=DataType.BINARY,
                target_resolution=TargetResolution.SEQUENCE,
                value_key="task_aligned_intervals.choice.choice",
                interval_key="task_aligned_intervals.choice",
                domain_key="task_aligned_intervals.choice",
                primary_metric="bacc",
                metrics={
                    "bacc": partial(Accuracy, task="multiclass", num_classes=2, average="macro"),
                    "f1": partial(F1Score, task="multiclass", num_classes=2, average="macro"),
                    "ap": partial(
                        AveragePrecision, task="multiclass", num_classes=2, average="macro"
                    ),
                },
            )
        case "reward":
            return TS1ReadoutSpec(
                id="reward",
                dim=2,
                data_type=DataType.BINARY,
                target_resolution=TargetResolution.SEQUENCE,
                value_key="task_aligned_intervals.reward.reward",
                interval_key="task_aligned_intervals.reward",
                domain_key="task_aligned_intervals.reward",
                primary_metric="bacc",
                metrics={
                    "bacc": partial(Accuracy, task="multiclass", num_classes=2, average="macro"),
                    "f1": partial(F1Score, task="multiclass", num_classes=2, average="macro"),
                    "ap": partial(
                        AveragePrecision, task="multiclass", num_classes=2, average="macro"
                    ),
                },
            )
        case "stimulus_contrast":
            return TS1ReadoutSpec(
                id="stimulus_contrast",
                dim=5,
                data_type=DataType.MULTINOMIAL,
                target_resolution=TargetResolution.SEQUENCE,
                value_key="task_aligned_intervals.stimulus_contrast.stimulus_contrast",
                interval_key="task_aligned_intervals.stimulus_contrast",
                domain_key="task_aligned_intervals.stimulus_contrast",
                primary_metric="bacc",
                metrics={
                    "bacc": partial(Accuracy, task="multiclass", num_classes=5, average="macro"),
                    "f1": partial(F1Score, task="multiclass", num_classes=5, average="macro"),
                    "ap": partial(
                        AveragePrecision, task="multiclass", num_classes=5, average="macro"
                    ),
                },
            )

        # regression tasks
        case "whisker_motion_energy":
            return TS1ReadoutSpec(
                id="whisker_motion_energy",
                dim=1,
                data_type=DataType.CONTINUOUS,
                target_resolution=TargetResolution.TIMESTEP,
                timestamp_key="whisker.timestamps",
                value_key="whisker.whisker_motion_energy",
                interval_key="task_aligned_intervals.movement_window",
                domain_key="whisker._domain",
                primary_metric="r2",
                metrics={
                    "r2": R2Score,
                    "mae": MeanAbsoluteError,
                    "pearson": partial(PearsonCorrCoef, num_outputs=1),
                },
            )
        case "wheel_speed":
            return TS1ReadoutSpec(
                id="wheel_speed",
                dim=1,
                data_type=DataType.CONTINUOUS,
                target_resolution=TargetResolution.TIMESTEP,
                timestamp_key="wheel.timestamps",
                value_key="wheel.wheel_speed",
                interval_key="task_aligned_intervals.movement_window",
                domain_key="wheel._domain",
                primary_metric="r2",
                metrics={
                    "r2": R2Score,
                    "mae": MeanAbsoluteError,
                    "pearson": partial(PearsonCorrCoef, num_outputs=1),
                },
            )
        case "right_paw_speed":
            return TS1ReadoutSpec(
                id="right_paw_speed",
                dim=1,
                data_type=DataType.CONTINUOUS,
                target_resolution=TargetResolution.TIMESTEP,
                timestamp_key="paws.timestamps",
                value_key="paws.right_paw_speed",
                mask_key="paws.is_right_paw_confident",
                mask_counts_key="task_aligned_intervals.movement_window.num_right_paw_confident",
                interval_key="task_aligned_intervals.movement_window",
                domain_key="paws._domain",
                primary_metric="r2",
                metrics={
                    "r2": R2Score,
                    "mae": MeanAbsoluteError,
                    "pearson": partial(PearsonCorrCoef, num_outputs=1),
                },
            )
        case "left_paw_speed":
            return TS1ReadoutSpec(
                id="left_paw_speed",
                dim=1,
                data_type=DataType.CONTINUOUS,
                target_resolution=TargetResolution.TIMESTEP,
                timestamp_key="paws.timestamps",
                value_key="paws.left_paw_speed",
                mask_key="paws.is_left_paw_confident",
                mask_counts_key="task_aligned_intervals.movement_window.num_left_paw_confident",
                interval_key="task_aligned_intervals.movement_window",
                domain_key="paws._domain",
                primary_metric="r2",
                metrics={
                    "r2": R2Score,
                    "mae": MeanAbsoluteError,
                    "pearson": partial(PearsonCorrCoef, num_outputs=1),
                },
            )
        case "licking_rate":
            return TS1ReadoutSpec(
                id="licking_rate",
                dim=1,
                data_type=DataType.EVENT_RATE,
                target_resolution=TargetResolution.TIMESTEP,
                timestamp_key="licks.timestamps",
                value_key="licks.licking_rate",
                interval_key="task_aligned_intervals.licking_window",
                domain_key="licks._domain",
                primary_metric="poisson_d2",
                metrics={
                    "poisson_d2": PoissonD2Score,
                    "mae": MeanAbsoluteError,
                },
            )

        case _:
            raise ValueError(
                f"{task!r} is not a scored TS1 task, expected one of {get_args(TS1Task)}"
            )


def get_ts1_supported_tasks(target_resolution: TargetResolution | None = None) -> list[TS1Task]:
    """Returns a list of all supported tasks.

    The items are TS1Task enum values, each with a string value naming the task.

    Returns:
        list[BenchmarkTasks]: A list of all supported tasks.

    Example:
        >>> tasks = get_ts1_supported_tasks()
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
