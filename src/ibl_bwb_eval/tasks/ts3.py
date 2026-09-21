"""TS3: the task vocabulary, and what each task predicts.

The multi-unit readout that pools these probabilities across neighbouring units lives in
``ibl_bwb_eval.multi_unit``, kept out of here so the task vocabulary stays numpy-free.
"""

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Literal, TypeAlias, get_args

from ibl_bwb_eval.tasks.types import TargetResolution

# Other entities (channel) and targets (ccf) are reserved in prds/ts3-extension-axes.md.
TS3Task: TypeAlias = Literal["unit_cosmos"]
"""The scored TS3 tasks, named ``<entity>_<target>``: what is classified, at which atlas level."""

# The Allen CCF Cosmos-level regions TS3 classifies, alphabetical: the order both
# sklearn's `classes_` and a plain `sorted()` produce, so submissions written before this
# was declared stay bit-compatible. `void` and `root` are dropped upstream and unscored.
COSMOS_LABELS: tuple[str, ...] = (
    "CB",
    "CNU",
    "CTXsp",
    "HB",
    "HPF",
    "HY",
    "Isocortex",
    "MB",
    "OLF",
    "TH",
)


# How a classified prediction is judged, per class and macro-averaged over classes. The
# names only: this module is stdlib-only, so the callables that compute them live in
# ``ibl_bwb_eval.metrics.classification`` and a test binds the two together. A task that is
# not classification (regressing a CCF coordinate, say) declares its own set instead.
CLASSIFICATION_METRICS: tuple[str, ...] = ("precision", "recall", "f1-score")


@dataclass
class TS3ReadoutSpec:
    """What one TS3 task predicts.

    Attributes:
        id: the task name.
        label_names: the class order every ``pred_proba`` column is indexed by. The scorer
            rejects a submission whose ``label_names`` disagree with the ground truth's,
            so this is part of the on-disk format.
        region_key: the ``units`` field the targets are read from.
        metrics: mapping from metric name to the callable that computes it. Per task,
            not per suite: region classification reports P/R/F1, and a task regressing a
            CCF coordinate would declare error metrics instead. Each callable takes
            ``(y_true, y_pred, label_names)`` and returns the scored names it owns.
    """

    id: TS3Task
    label_names: tuple[str, ...]
    region_key: str
    metrics: dict[str, Callable]
    target_resolution: TargetResolution = TargetResolution.SEQUENCE

    @property
    def dim(self) -> int:
        """Number of classes. Derived, so it cannot drift from ``label_names``."""
        return len(self.label_names)

    @property
    def num_timesteps(self) -> int:
        """Always one: a TS3 target is a unit's label, not a time course."""
        return 1

    def score(self, y_true, y_pred) -> dict[str, float]:
        """Every metric this task declares, flattened.

        The scorer and the probes both call this, so a run's live scores and its offline
        scores cannot diverge.
        """
        out: dict[str, float] = {}
        for fn in self.metrics.values():
            out.update(fn(y_true, y_pred, self.label_names))
        return out


def get_ts3_readout_spec(task: TS3Task) -> TS3ReadoutSpec:
    """What ``task`` classifies, its label vocabulary, and the metrics it is scored on."""
    from ibl_bwb_eval.metrics import CLASSIFICATION_SCORERS

    match task:
        case "unit_cosmos":
            return TS3ReadoutSpec(
                id="unit_cosmos",
                label_names=COSMOS_LABELS,
                region_key="region_cosmos",
                metrics=CLASSIFICATION_SCORERS,
            )
        case _:
            raise ValueError(
                f"{task!r} is not a scored TS3 task, expected one of {get_args(TS3Task)}"
            )


def check_ts3_label_order(labels: Sequence[str], task: TS3Task) -> None:
    """Raise unless ``labels`` is the task's vocabulary, in order.

    Call this wherever a class order is derived from data rather than taken from the
    spec: a silent disagreement misindexes every probability column.
    """
    expected = get_ts3_readout_spec(task).label_names
    got = tuple(str(x) for x in labels)
    if got != expected:
        raise ValueError(
            f"Region order does not match the {task!r} vocabulary.\n"
            f"  expected: {expected}\n"
            f"  got:      {got}"
        )
