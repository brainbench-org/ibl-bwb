"""The task vocabulary of the three suites, and the task id used in submission paths.

Each suite's module here declares its own tasks and what they predict, so adding a task
touches one file. The three readout spec types are deliberately not unified: TS1's is
trial-aligned (interval and value keys, a target resolution) and TS3's is unit-level (an ordered
label vocabulary), and they share only the slice :class:`ReadoutSpec` names. TS2's sizes no
readout head at all, and carries the mask axis and the metrics instead. Every readout name
carries its suite, the way ``TS1Task`` and ``TS3Task`` already do, so the three sets can be
exported side by side here and a call site says which suite it means without being read in
context.

A prediction file is filed under ``{label}/{task_id}/...``, and the scorers select the
files they own by that id, so the ``tsN-`` prefix is part of the on-disk format rather
than a display string. :func:`task_id` writes it and :func:`is_task_of` reads it: they
are the only two places that know its shape.

Everything reachable from here is stdlib-only, so reading the task list costs no numpy
or torch import. That is why ``multi_unit_prediction`` lives in ``ibl_bwb_eval.multi_unit``.
"""

from typing import Literal, TypeAlias, get_args

from ibl_bwb_eval.tasks import ts1, ts2, ts3
from ibl_bwb_eval.tasks.ts1 import (
    BEHAVIOR_SFREQ,
    TARGET_WINDOW,
    TS1ReadoutSpec,
    TS1Task,
    get_ts1_readout_spec,
    get_ts1_supported_tasks,
)
from ibl_bwb_eval.tasks.ts2 import TS2ReadoutSpec, TS2Task, get_ts2_readout_spec
from ibl_bwb_eval.tasks.ts3 import (
    CLASSIFICATION_METRICS,
    COSMOS_LABELS,
    TS3ReadoutSpec,
    TS3Task,
    check_ts3_label_order,
    get_ts3_readout_spec,
)
from ibl_bwb_eval.tasks.types import DataType, ReadoutSpec, TargetResolution

Suite: TypeAlias = Literal["ts1", "ts2", "ts3"]

SUITE_TASKS: dict[Suite, tuple[str, ...]] = {
    "ts1": get_args(TS1Task),
    "ts2": get_args(TS2Task),
    "ts3": get_args(TS3Task),
}


def task_id(suite: Suite, task: str) -> str:
    """The flattened task id a submission is filed under, e.g. ``"ts2-co_smoothing"``."""
    if task not in SUITE_TASKS[suite]:
        raise ValueError(f"{task!r} is not a {suite} task, expected one of {SUITE_TASKS[suite]}")
    return f"{suite}-{task}"


def is_task_of(suite: Suite, task_id: str) -> bool:
    """Whether a flattened task id belongs to ``suite``. The scorers' file filter."""
    return task_id.startswith(f"{suite}-")


def task_of(suite: Suite, task_id: str) -> str:
    """The bare task name inside a flattened id. The inverse of :func:`task_id`."""
    if not is_task_of(suite, task_id):
        raise ValueError(f"{task_id!r} is not a {suite} task")
    return task_id.removeprefix(f"{suite}-")


__all__ = [
    "BEHAVIOR_SFREQ",
    "CLASSIFICATION_METRICS",
    "COSMOS_LABELS",
    "SUITE_TASKS",
    "TARGET_WINDOW",
    "DataType",
    "ReadoutSpec",
    "Suite",
    "TS1ReadoutSpec",
    "TS1Task",
    "TS2ReadoutSpec",
    "TS2Task",
    "TS3ReadoutSpec",
    "TS3Task",
    "TargetResolution",
    "check_ts3_label_order",
    "get_ts1_readout_spec",
    "get_ts1_supported_tasks",
    "get_ts2_readout_spec",
    "get_ts3_readout_spec",
    "is_task_of",
    "task_id",
    "task_of",
    "ts1",
    "ts2",
    "ts3",
]
