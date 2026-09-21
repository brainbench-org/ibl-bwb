"""``context_length`` is checked twice: the base class bounds it against
``MAX_CONTEXT_LENGTH``, and a suite (TS1, TS2) additionally bounds it below by its own
``TARGET_WINDOW``, since the base class no longer knows what any suite's scored window
is. Both checks run before any disk access, so a caller learns the window is invalid
before finding out its data does not exist.

The base class's check needs nothing but a bogus ``root``: it runs before
``read_recording_ids`` or any other I/O (see ``test_splits.py`` for the established
style of testing this repo's datasets without real data). A suite-level check runs
*after* ``super().__init__()`` returns, and that call loads real recording files, so
reaching it requires a real, vendored recording id (``read_recording_ids`` reads a
checked-in list, no data root needed) *and* stubbing out the two steps that would
otherwise need an actual data build: ``torch_brain.datasets.dataset.Dataset.__init__``
(which checks HDF5 files exist) and ``core.dataset.read_build_unit_filtering`` (TS2
only, since it declares ``require_unit_filtering``). Neither stub touches the
``context_length`` logic itself.

Only TS1 and TS2 expose ``context_length``. It's an eval-pipeline capability, not a
pretraining one; no pretrain dataset or trainer wires it in.
"""

from unittest.mock import MagicMock

import pytest
from torch_brain.datasets.dataset import Dataset as TorchBrainDataset

from core.data import read_recording_ids
from ibl_bwb_eval.tasks.ts1 import TARGET_WINDOW as TS1_TARGET_WINDOW
from ibl_bwb_eval.tasks.ts2 import COSMOOTH_TARGET_WINDOW as TS2_TARGET_WINDOW
from ts1.ts1_dataset import IBLBrainWideBenchTS1
from ts2.ts2_dataset import IBLBrainWideBenchTS2

MAX_CONTEXT_LENGTH = 20.0

EVAL_RECORDING_ID = read_recording_ids("eval")[0]


@pytest.fixture
def stub_disk_io(monkeypatch):
    """Let a dataset's ``__init__`` run to completion without a real data build.

    Patches the grandparent ``Dataset.__init__`` (HDF5 file existence + loading) to a
    no-op that only sets ``recording_ids``, and ``read_build_unit_filtering`` (the one
    other place disk is touched during construction, for a class that passes
    ``require_unit_filtering``) to return a permissive stand-in.
    """

    def fake_dataset_init(self, **kwargs):
        self._recording_ids = sorted(kwargs.get("recording_ids") or [])

    monkeypatch.setattr(TorchBrainDataset, "__init__", fake_dataset_init)
    monkeypatch.setattr("core.dataset.enforce_unit_filtering", lambda *a, **kw: None)
    monkeypatch.setattr("core.dataset.read_build_unit_filtering", lambda self: MagicMock())


def _build(cls, **kwargs):
    return cls(root="/nonexistent-fake-root", **kwargs)


# ----------------------------------------------------------------------------------
# Base class: context_length <= MAX_CONTEXT_LENGTH. No stubbing needed, this fires
# before read_recording_ids / any I/O, same style as test_splits.py.
# ----------------------------------------------------------------------------------

UPPER_BOUND_CASES = [
    (IBLBrainWideBenchTS1, {"split": "test", "task": "choice", "recording_id": "fake"}),
    (IBLBrainWideBenchTS2, {"split": "test", "task": "co_smoothing", "recording_id": "fake"}),
]


@pytest.mark.parametrize(
    "cls,kwargs", UPPER_BOUND_CASES, ids=[c.__name__ for c, _ in UPPER_BOUND_CASES]
)
def test_context_length_above_the_max_is_refused_before_any_io(cls, kwargs):
    with pytest.raises(AssertionError, match="MAX_CONTEXT_LENGTH"):
        _build(cls, context_length=MAX_CONTEXT_LENGTH + 5.0, **kwargs)


@pytest.mark.parametrize(
    "cls,kwargs", UPPER_BOUND_CASES, ids=[c.__name__ for c, _ in UPPER_BOUND_CASES]
)
def test_context_length_at_the_max_is_not_refused_by_the_bound_check(cls, kwargs):
    """Boundary-valid: whatever fails next is disk I/O, not the context_length check."""
    with pytest.raises(Exception) as exc_info:
        _build(cls, context_length=MAX_CONTEXT_LENGTH, **kwargs)
    assert "MAX_CONTEXT_LENGTH" not in str(exc_info.value)
    assert "TARGET_WINDOW" not in str(exc_info.value)


# ----------------------------------------------------------------------------------
# Suite-level lower bound: context_length >= TARGET_WINDOW. Runs after
# super().__init__() returns, so it needs the disk stubs and a real recording id.
# ----------------------------------------------------------------------------------

LOWER_BOUND_CASES = [
    (
        IBLBrainWideBenchTS1,
        TS1_TARGET_WINDOW,
        {"split": "test", "task": "choice", "recording_id": EVAL_RECORDING_ID},
    ),
    (
        IBLBrainWideBenchTS2,
        TS2_TARGET_WINDOW,
        {"split": "test", "recording_id": EVAL_RECORDING_ID, "task": "co_smoothing"},
    ),
]


@pytest.mark.parametrize(
    "cls,target_window,kwargs", LOWER_BOUND_CASES, ids=[c.__name__ for c, _, _ in LOWER_BOUND_CASES]
)
def test_context_length_below_target_window_is_refused(cls, target_window, kwargs, stub_disk_io):
    with pytest.raises(AssertionError, match="TARGET_WINDOW"):
        _build(cls, context_length=target_window / 2, **kwargs)


@pytest.mark.parametrize(
    "cls,target_window,kwargs", LOWER_BOUND_CASES, ids=[c.__name__ for c, _, _ in LOWER_BOUND_CASES]
)
@pytest.mark.parametrize(
    "context_length", [None, 5.0, MAX_CONTEXT_LENGTH], ids=["at_target_window", "mid", "at_max"]
)
def test_context_length_from_target_window_to_max_is_accepted(
    cls, target_window, kwargs, stub_disk_io, context_length
):
    """Every valid point (the lower bound, an interior value, the upper bound) must
    fully construct: no assertion left to trip once both bounds hold."""
    if context_length is None:  # "at_target_window": the lower bound itself
        context_length = target_window
    obj = _build(cls, context_length=context_length, **kwargs)
    assert obj.context_window == context_length
