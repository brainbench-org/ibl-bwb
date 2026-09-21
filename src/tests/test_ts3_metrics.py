"""TS3 metrics: what the spec declares, and that moving them off classification_report
did not move a single reported number.

Both scoring paths used to build one ``classification_report`` and pick columns out of it,
so anything outside precision/recall/f1-score raised KeyError and a non-classification task
was unreachable. They now call the callables the spec declares. The metric names are what
``SELECTION_METRIC`` and the aggregation key on, so they are pinned here.
"""

import numpy as np
import pytest
from sklearn.metrics import classification_report

from ibl_bwb_eval.metrics.classification import CLASSIFICATION_SCORERS
from ibl_bwb_eval.tasks import CLASSIFICATION_METRICS, COSMOS_LABELS, get_ts3_readout_spec

SPEC = get_ts3_readout_spec("unit_cosmos")


def _scores(y_true, y_pred) -> dict[str, float]:
    return SPEC.score(y_true, y_pred)


def _sample(n=400, seed=0):
    rng = np.random.default_rng(seed)
    names = np.array(COSMOS_LABELS)
    return names[rng.integers(0, len(names), n)], names[rng.integers(0, len(names), n)]


def test_the_stdlib_vocabulary_and_the_scorers_agree():
    """tasks/ts3.py is stdlib-only and cannot import the callables, so bind them here."""
    assert tuple(CLASSIFICATION_SCORERS) == CLASSIFICATION_METRICS
    assert tuple(SPEC.metrics) == CLASSIFICATION_METRICS


def test_metric_names_are_pinned():
    """These are the keys SELECTION_METRIC and the aggregation read."""
    got = set(_scores(*_sample()))
    expected = {f"{c}/{m}" for c in COSMOS_LABELS for m in CLASSIFICATION_METRICS}
    expected |= {f"macro/{m}" for m in CLASSIFICATION_METRICS}
    assert got == expected


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_values_match_the_classification_report_they_replaced(seed):
    y_true, y_pred = _sample(seed=seed)
    report = classification_report(
        y_true, y_pred, labels=list(COSMOS_LABELS), output_dict=True, zero_division=0
    )
    got = _scores(y_true, y_pred)

    for c in COSMOS_LABELS:
        for m in CLASSIFICATION_METRICS:
            assert got[f"{c}/{m}"] == pytest.approx(report[c][m]), f"{c}/{m}"
    for m in CLASSIFICATION_METRICS:
        assert got[f"macro/{m}"] == pytest.approx(report["macro avg"][m]), f"macro/{m}"


def test_a_class_missing_from_both_sides_scores_zero():
    """The old scorer passed no labels=, so an unseen class dropped out and KeyError'd."""
    present = [c for c in COSMOS_LABELS if c != "CTXsp"]
    y = np.array(present)
    got = _scores(y, y)
    assert got["CTXsp/f1-score"] == 0.0
    assert got["CB/f1-score"] == 1.0
