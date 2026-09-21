"""The task vocabulary and the submission task id.

``task_id`` names the directory a submission is filed under and the scorers select
their files by it, so its exact spelling is part of the published format: changing the
separator or a task name silently orphans every prediction file already written.
"""

import pytest

from ibl_bwb_eval.tasks import COSMOS_LABELS, SUITE_TASKS, get_ts3_readout_spec, is_task_of, task_id


def test_task_id_spelling_is_pinned():
    """Literal expectations, so a refactor of task_id cannot change the on-disk names."""
    assert task_id("ts1", "choice") == "ts1-choice"
    assert task_id("ts2", "co_smoothing") == "ts2-co_smoothing"
    assert task_id("ts3", "unit_cosmos") == "ts3-unit_cosmos"


def test_declared_tasks_are_pinned():
    assert SUITE_TASKS["ts1"] == (
        "choice",
        "reward",
        "stimulus_contrast",
        "whisker_motion_energy",
        "wheel_speed",
        "right_paw_speed",
        "left_paw_speed",
        "licking_rate",
    )
    assert SUITE_TASKS["ts2"] == ("co_smoothing", "forecasting")
    assert SUITE_TASKS["ts3"] == ("unit_cosmos",)


def test_cosmos_label_order_is_pinned():
    """The column order of every TS3 pred_proba, and the label_names the scorer matches on."""
    assert COSMOS_LABELS == (
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
    assert get_ts3_readout_spec("unit_cosmos").label_names == COSMOS_LABELS
    assert get_ts3_readout_spec("unit_cosmos").dim == 10


def test_ts3_readout_spec_rejects_an_unscored_task():
    with pytest.raises(ValueError, match="not a scored TS3 task"):
        get_ts3_readout_spec("unit_beryl")


@pytest.mark.parametrize("suite", sorted(SUITE_TASKS))
def test_every_task_round_trips_through_its_own_suite_only(suite):
    others = [s for s in SUITE_TASKS if s != suite]
    for task in SUITE_TASKS[suite]:
        tid = task_id(suite, task)
        assert is_task_of(suite, tid)
        assert not any(is_task_of(other, tid) for other in others)


def test_task_id_rejects_a_task_the_suite_does_not_have():
    with pytest.raises(ValueError, match="not a ts2 task"):
        task_id("ts2", "choice")
