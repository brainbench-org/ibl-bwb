"""TS2: the task vocabulary, and what each task predicts."""

from collections.abc import Callable
from dataclasses import dataclass
from typing import Literal, TypeAlias, get_args

TS2Task: TypeAlias = Literal[
    "co_smoothing",
    "forecasting",
]
"""The two scored TS2 tasks, holding out units and trailing timesteps respectively."""

# Co-smoothing's scored window: the trailing second, regardless of context length.
COSMOOTH_TARGET_WINDOW = 1.0  # seconds

# Forecasting's reference window: the lower bound on context_length and the base
# that IBLBrainWideBenchTS2's context extension anchors to. The actual scored region
# is FORECAST_RATIO of it (see IBLBrainWideBenchTS2.FORECAST_RATIO), fixed
# regardless of context_length.
FORECAST_BASE_WINDOW = 1.0  # seconds


@dataclass
class TS2ReadoutSpec:
    """What one TS2 task predicts.

    Attributes:
        id: the task name.
        mask_dim: the dim of a ``(batch, time, units)`` tensor the held-out mask varies
            along. Co-smoothing holds out units, forecasting holds out timesteps; the
            mask is constant along every other dim.
        primary_metric: the metric used for model selection and for ``best/*/avg``.
        metrics: mapping from metric name to a callable that instantiates the metric.
    """

    id: TS2Task
    mask_dim: int
    primary_metric: str
    metrics: dict[str, Callable]


def get_ts2_readout_spec(task: TS2Task) -> TS2ReadoutSpec:
    """What ``task`` holds out, and the metrics it is scored on."""
    from ibl_bwb_eval.metrics import BPS, PoissonD2Score

    match task:
        case "co_smoothing":
            return TS2ReadoutSpec(
                id="co_smoothing",
                mask_dim=2,  # held-out units
                primary_metric="poisson_d2",
                metrics={"poisson_d2": PoissonD2Score, "bps": BPS},
            )
        case "forecasting":
            return TS2ReadoutSpec(
                id="forecasting",
                mask_dim=1,  # held-out timesteps
                primary_metric="poisson_d2",
                metrics={"poisson_d2": PoissonD2Score, "bps": BPS},
            )
        case _:
            raise ValueError(
                f"{task!r} is not a scored TS2 task, expected one of {get_args(TS2Task)}"
            )
