"""Tests for ibl_bwb_eval.scoring.aggregation: clip-then-average ordering, TS3 normalization,
and the anchored significance-based ranking."""

import statistics
import warnings

import pytest

from ibl_bwb_eval.scoring.aggregation import (
    _assign_ranks_anchored,
    _is_significantly_better,
    aggregate,
    aggregate_over_seeds,
    from_ts3,
    rank,
)


def test_from_ts3_injects_task_into_session_less_keys():
    raw = {("modelA", 0): {"macro/f1-score": 0.8}, ("modelA", 1): {"macro/f1-score": 0.9}}
    normalized = from_ts3(raw, task="ts3-unit_cosmos")
    assert normalized == {
        ("modelA", "ts3-unit_cosmos", 0): {"macro/f1-score": 0.8},
        ("modelA", "ts3-unit_cosmos", 1): {"macro/f1-score": 0.9},
    }


def test_aggregate_accepts_3tuple_keys_without_recording_id():
    # a DB-backed caller's own TS3 storage, with no recording_id column at all --
    # no from_ts3 or other explicit normalization step required.
    raw = {
        ("modelA", "ts3-unit_cosmos", 0): {"macro/f1-score": 0.60},
        ("modelA", "ts3-unit_cosmos", 1): {"macro/f1-score": 0.64},
    }
    summary = aggregate(raw)
    assert set(summary) == {("modelA", "ts3-unit_cosmos", "__all__")}
    mean, _sem, n = summary[("modelA", "ts3-unit_cosmos", "__all__")]["macro/f1-score"]
    assert mean == pytest.approx(0.62)
    assert n == 2


def test_aggregate_mixed_3tuple_and_4tuple_keys():
    raw = {
        ("modelA", "ts1-choice", "rec1", 0): {"bacc": 0.6},
        ("modelA", "ts3-unit_cosmos", 0): {"macro/f1-score": 0.6},
    }
    summary = aggregate(raw)
    assert set(summary) == {
        ("modelA", "ts1-choice", "rec1"),
        ("modelA", "ts3-unit_cosmos", "__all__"),
    }


def test_aggregate_rejects_malformed_key():
    with pytest.raises(ValueError, match="3- or 4-tuple"):
        aggregate({("modelA", "ts3-unit_cosmos"): {"macro/f1-score": 0.6}})


def test_aggregate_clips_before_averaging_not_after():
    """Seeds straddling zero must be clipped individually, then averaged -- not
    averaged then clipped. clip([-0.5, 0.5]) -> mean 0.25; naive avg-then-clip would
    give clip(mean(-0.5, 0.5)) = clip(0) = 0.
    """
    raw = {
        ("modelA", "ts2-co_smoothing", "rec1", 0): {"poisson_d2": -0.5},
        ("modelA", "ts2-co_smoothing", "rec1", 1): {"poisson_d2": 0.5},
    }
    summary = aggregate(raw)
    mean, _sem, n = summary[("modelA", "ts2-co_smoothing", "rec1")]["poisson_d2"]
    assert mean == pytest.approx(0.25)
    assert n == 2


def test_aggregate_only_clips_designated_metrics():
    raw = {
        ("modelA", "ts1-choice", "rec1", 0): {"bacc": -0.1, "r2": -0.1},
        ("modelA", "ts1-choice", "rec1", 1): {"bacc": -0.3, "r2": -0.3},
    }
    summary = aggregate(raw)
    metrics = summary[("modelA", "ts1-choice", "rec1")]
    assert metrics["bacc"][0] == pytest.approx(-0.2)  # unclipped, can go negative
    assert metrics["r2"][0] == 0.0  # clip(-0.1) = clip(-0.3) = 0 -> mean 0


def test_aggregate_clips_bps_like_poisson_d2():
    # bps is null-relative like poisson_d2 (negative just means worse than null),
    # so it's clipped the same way.
    raw = {
        ("modelA", "ts2-forecasting", "rec1", 0): {"bps": -0.1},
        ("modelA", "ts2-forecasting", "rec1", 1): {"bps": -0.3},
    }
    summary = aggregate(raw)
    mean, _sem, _n = summary[("modelA", "ts2-forecasting", "rec1")]["bps"]
    assert mean == 0.0


def test_aggregate_groups_by_label_task_recording_not_seed():
    raw = {
        ("modelA", "ts1-choice", "rec1", 0): {"bacc": 0.6},
        ("modelA", "ts1-choice", "rec1", 1): {"bacc": 0.8},
        ("modelA", "ts1-choice", "rec2", 0): {"bacc": 0.4},
        ("modelB", "ts1-choice", "rec1", 0): {"bacc": 0.5},
    }
    summary = aggregate(raw)
    assert set(summary) == {
        ("modelA", "ts1-choice", "rec1"),
        ("modelA", "ts1-choice", "rec2"),
        ("modelB", "ts1-choice", "rec1"),
    }
    mean, sem, n = summary[("modelA", "ts1-choice", "rec1")]["bacc"]
    assert mean == pytest.approx(0.7)
    assert n == 2
    assert sem is not None
    mean, sem, n = summary[("modelB", "ts1-choice", "rec1")]["bacc"]
    assert n == 1
    assert sem is None


def test_is_significantly_better_requires_higher_mean():
    assert not _is_significantly_better(0.5, 0.01, 5, 0.9, 0.01, 5, alpha=0.05)


def test_is_significantly_better_clear_separation():
    assert _is_significantly_better(0.9, 0.01, 5, 0.5, 0.01, 5, alpha=0.05)


def test_is_significantly_better_overlapping_not_significant():
    assert not _is_significantly_better(0.51, 0.2, 5, 0.50, 0.2, 5, alpha=0.05)


def test_is_significantly_better_single_seed_trusts_point_estimate():
    assert _is_significantly_better(0.9, None, 1, 0.5, None, 1, alpha=0.05)


def test_assign_ranks_anchored_clear_winner_then_tie():
    # A clearly best; B and C statistically tied for second.
    means = [0.9, 0.5, 0.51]
    sems = [0.01, 0.02, 0.02]
    ns = [30, 30, 30]
    ranks = _assign_ranks_anchored(means, sems, ns, alpha=0.05)
    assert ranks[0] == 1  # A
    assert ranks[1] == ranks[2] == 2  # B, C tied for 2nd (competition ranking)


def test_assign_ranks_anchored_all_distinct():
    means = [0.9, 0.5, 0.1]
    sems = [0.001, 0.001, 0.001]
    ns = [5, 5, 5]
    ranks = _assign_ranks_anchored(means, sems, ns, alpha=0.05)
    assert ranks == [1, 2, 3]


def test_assign_ranks_naive_ignores_overlapping_sems():
    # Same B/C means as the step_down "tied" case, but naive ranks strictly by mean.
    means = [0.9, 0.5, 0.51]
    sems = [0.01, 0.02, 0.02]
    ns = [30, 30, 30]
    ranks = _assign_ranks_anchored(means, sems, ns, alpha=0.05, method="naive")
    assert ranks == [1, 3, 2]  # A, C, B strictly by mean -- no ties


def test_assign_ranks_naive_ties_only_on_exact_equality():
    means = [0.9, 0.5, 0.5]
    sems = [0.01, 0.02, 0.02]
    ns = [30, 30, 30]
    ranks = _assign_ranks_anchored(means, sems, ns, alpha=0.05, method="naive")
    assert ranks[0] == 1
    assert ranks[1] == ranks[2] == 2


def test_assign_ranks_unknown_method_raises():
    with pytest.raises(ValueError, match="Unknown ranking method"):
        _assign_ranks_anchored([0.9, 0.5], [0.01, 0.01], [5, 5], alpha=0.05, method="bogus")


def test_rank_naive_method_threaded_through():
    summary = {
        ("modelA", "ts1-choice", "rec1"): {"bacc": (0.9, 0.01, 30)},
        ("modelB", "ts1-choice", "rec1"): {"bacc": (0.5, 0.02, 30)},
        ("modelC", "ts1-choice", "rec1"): {"bacc": (0.51, 0.02, 30)},
    }
    step_down = rank(summary, primary_metric={"ts1-choice": "bacc"}, method="step_down")
    naive = rank(summary, primary_metric={"ts1-choice": "bacc"}, method="naive")
    assert step_down["ts1-choice"]["modelB"] == step_down["ts1-choice"]["modelC"]  # tied
    assert naive["ts1-choice"]["modelB"] != naive["ts1-choice"]["modelC"]  # strict order


def test_aggregate_over_seeds_averages_recordings_then_takes_sem_over_seeds():
    # two recordings x three seeds. Per seed the recording-average is 0.8, 0.6, 0.4,
    # so the reported SEM is the spread of those three, not anything per-recording.
    raw = {
        ("modelA", "ts1-choice", "rec1", 0): {"bacc": 0.9},
        ("modelA", "ts1-choice", "rec2", 0): {"bacc": 0.7},
        ("modelA", "ts1-choice", "rec1", 1): {"bacc": 0.7},
        ("modelA", "ts1-choice", "rec2", 1): {"bacc": 0.5},
        ("modelA", "ts1-choice", "rec1", 2): {"bacc": 0.5},
        ("modelA", "ts1-choice", "rec2", 2): {"bacc": 0.3},
    }
    mean, sem, n = aggregate_over_seeds(raw)[("modelA", "ts1-choice")]["bacc"]
    assert mean == pytest.approx(0.6)
    assert n == 3  # seeds, not recordings
    assert sem == pytest.approx(statistics.stdev([0.8, 0.6, 0.4]) / 3**0.5)


def test_aggregate_over_seeds_clips_before_averaging_recordings():
    # -0.4 clips to 0 per (recording, seed) first, so seed 0 averages (0 + 0.6) / 2.
    # Clipping the recording-average instead would give max((-0.4 + 0.6) / 2, 0) = 0.1.
    raw = {
        ("modelA", "ts1-wheel_speed", "rec1", 0): {"r2": -0.4},
        ("modelA", "ts1-wheel_speed", "rec2", 0): {"r2": 0.6},
        ("modelA", "ts1-wheel_speed", "rec1", 1): {"r2": 0.2},
        ("modelA", "ts1-wheel_speed", "rec2", 1): {"r2": 0.4},
    }
    mean, _sem, _n = aggregate_over_seeds(raw)[("modelA", "ts1-wheel_speed")]["r2"]
    assert mean == pytest.approx(0.3)  # mean of 0.3 and 0.3, not of 0.1 and 0.3


def test_aggregate_over_seeds_sem_is_none_for_a_single_seed():
    raw = {("modelA", "ts1-choice", "rec1", 0): {"bacc": 0.9}}
    _mean, sem, n = aggregate_over_seeds(raw)[("modelA", "ts1-choice")]["bacc"]
    assert sem is None
    assert n == 1


def test_aggregate_over_seeds_skips_the_recording_average_for_ts3_shaped_keys():
    # TS3 has no session dimension, so the 3-tuple keys pass straight to the seed stage.
    raw = {
        ("modelA", "ts3-unit_cosmos", 0): {"macro/f1-score": 0.62},
        ("modelA", "ts3-unit_cosmos", 1): {"macro/f1-score": 0.64},
    }
    mean, sem, n = aggregate_over_seeds(raw)[("modelA", "ts3-unit_cosmos")]["macro/f1-score"]
    assert mean == pytest.approx(0.63)
    assert n == 2
    assert sem == pytest.approx(statistics.stdev([0.62, 0.64]) / 2**0.5)


def test_rank_averages_over_recording_id_within_task():
    summary = {
        ("modelA", "ts1-choice", "rec1"): {"bacc": (0.9, 0.01, 5)},
        ("modelB", "ts1-choice", "rec1"): {"bacc": (0.5, 0.01, 5)},
        ("modelA", "ts1-choice", "rec2"): {"bacc": (0.5, 0.01, 5)},
        ("modelB", "ts1-choice", "rec2"): {"bacc": (0.9, 0.01, 5)},
    }
    result = rank(summary, primary_metric={"ts1-choice": "bacc"})
    # each model wins exactly one of the two sessions -> avg rank 1.5 for both
    assert result["ts1-choice"]["modelA"] == pytest.approx(1.5)
    assert result["ts1-choice"]["modelB"] == pytest.approx(1.5)


def test_rank_is_dataframe_constructible_dict_of_dicts():
    summary = {
        ("modelA", "ts1-choice", "rec1"): {"bacc": (0.9, 0.01, 5)},
        ("modelB", "ts1-choice", "rec1"): {"bacc": (0.5, 0.01, 5)},
    }
    result = rank(summary, primary_metric={"ts1-choice": "bacc"})
    assert result == {"ts1-choice": {"modelA": 1.0, "modelB": 2.0}}
    pd = pytest.importorskip("pandas")
    df = pd.DataFrame(result)
    assert df.loc["modelA", "ts1-choice"] == 1.0


def test_rank_missing_primary_metric_raises():
    summary = {("modelA", "ts1-choice", "rec1"): {"f1": (0.9, 0.01, 5)}}
    with pytest.raises(KeyError):
        rank(summary, primary_metric={"ts1-choice": "bacc"})


def test_rank_warns_when_a_recording_has_only_one_label():
    # modelA and modelB share rec1, but modelB alone has rec2 -- no comparison there.
    summary = {
        ("modelA", "ts1-choice", "rec1"): {"bacc": (0.9, 0.01, 5)},
        ("modelB", "ts1-choice", "rec1"): {"bacc": (0.5, 0.01, 5)},
        ("modelB", "ts1-choice", "rec2"): {"bacc": (0.5, 0.01, 5)},
    }
    with pytest.warns(UserWarning, match="ts1-choice.*1/2 recording_id"):
        result = rank(summary, primary_metric={"ts1-choice": "bacc"})
    # ranking still proceeds despite the warning -- rec2's trivial rank of 1 still counts
    assert result["ts1-choice"]["modelB"] == pytest.approx(1.5)  # (2 + 1) / 2
    assert result["ts1-choice"]["modelA"] == pytest.approx(1.0)


def test_rank_no_warning_with_full_coverage():
    summary = {
        ("modelA", "ts1-choice", "rec1"): {"bacc": (0.9, 0.01, 5)},
        ("modelB", "ts1-choice", "rec1"): {"bacc": (0.5, 0.01, 5)},
    }
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        rank(summary, primary_metric={"ts1-choice": "bacc"})  # must not raise
