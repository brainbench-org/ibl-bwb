"""Regression tests against the v01 legacy prediction fixtures.

``fixtures/legacy/v01_ts0{1,2,3}.dat`` are frozen bytes written before
``PredictionsWriter`` stamped its own version. Predictions already on disk have that
shape, so the scorers have to keep reading it however the writer evolves, which a
roundtrip test cannot show since it writes and reads with the same code.
"""

from pathlib import Path

import numpy as np
import torch
from safetensors import safe_open
from safetensors.torch import save_file

from ibl_bwb_eval._version import __version__
from ibl_bwb_eval.predictions import PredictionsWriter, version_of
from ibl_bwb_eval.scoring import ts1 as ts1_scoring
from ibl_bwb_eval.scoring import ts2 as ts2_scoring
from ibl_bwb_eval.scoring import ts3 as ts3_scoring
from ibl_bwb_eval.tasks import COSMOS_LABELS

FIXTURES = Path(__file__).resolve().parent / "fixtures" / "legacy"


def test_v01_fixtures_carry_no_version():
    """A v01 file predates the stamp, which version_of reports as None."""
    for name in ("v01_ts01.dat", "v01_ts02.dat", "v01_ts03.dat"):
        with safe_open(str(FIXTURES / name), framework="pt") as f:
            meta = f.metadata()
            assert "ibl_bwb_eval_version" not in meta
            assert version_of(meta) is None


def test_version_of_current_write(tmp_path):
    """A file written by the current PredictionsWriter reports the current version."""
    writer = PredictionsWriter(
        enable=True, base_path=tmp_path, task="ts1-choice", seed=1, label="x"
    )
    writer.set(predictions=torch.randn(2, 1, 2), trial_id=torch.arange(2))
    writer.save()

    path = next(tmp_path.rglob("seed_*.safetensors"))
    with safe_open(str(path), framework="pt") as f:
        meta = f.metadata()
    assert version_of(meta) == __version__


def test_v01_ts1_scores(tmp_path):
    """A v01 ts1-choice file still scores against a matching ground truth."""
    pred_path = FIXTURES / "v01_ts01.dat"
    with safe_open(str(pred_path), framework="pt") as f:
        trial_id = f.get_tensor("trial_id")
        recording_id = f.metadata()["recording_id"]

    gt_path = tmp_path / "ground_truth.safetensors"
    save_file(
        {
            "values": torch.randint(0, 2, (len(trial_id), 1)).float(),
            "trial_id": trial_id,
            "timestamps": torch.zeros(len(trial_id), 1),
        },
        str(gt_path),
        metadata={"task": "ts1-choice", "recording_id": recording_id},
    )

    metrics = ts1_scoring.score_file("ts1-choice", pred_path, gt_path)
    assert {"bacc", "f1", "ap"} <= set(metrics)
    assert all(np.isfinite(v) for v in metrics.values())


def test_v01_ts2_scores(tmp_path):
    """A v01 ts2-co_smoothing file still scores against a matching ground truth."""
    pred_path = FIXTURES / "v01_ts02.dat"
    with safe_open(str(pred_path), framework="pt") as f:
        window_timestamps = f.get_tensor("window_timestamps")
        unit_ids = f.get_tensor("unit_ids")
        predictions_shape = f.get_tensor("predictions").shape

    gt_path = tmp_path / "ground_truth.safetensors"
    save_file(
        {
            "targets": torch.randint(0, 5, tuple(predictions_shape)).float(),
            "window_timestamps": window_timestamps,
            "unit_ids": unit_ids,
        },
        str(gt_path),
        metadata={"task": "ts2-co_smoothing"},
    )

    metrics = ts2_scoring.score_file(pred_path, gt_path)
    assert {"poisson_d2", "bps"} <= set(metrics)
    assert all(np.isfinite(v) for v in metrics.values())


def test_v01_ts3_scores(tmp_path):
    """A v01 ts3-unit_cosmos file still scores against a matching ground truth."""
    pred_path = FIXTURES / "v01_ts03.dat"
    with safe_open(str(pred_path), framework="pt") as f:
        entity_ids = f.get_tensor("entity_ids")
        label_names = f.metadata()["label_names"]

    c = len(COSMOS_LABELS)
    gt_path = tmp_path / "ground_truth.safetensors"
    save_file(
        {
            "targets": torch.randint(0, c, (len(entity_ids),)),
            "entity_ids": entity_ids,
        },
        str(gt_path),
        metadata={"label_names": label_names},
    )

    metrics = ts3_scoring.score_file(pred_path, gt_path)
    assert "macro/f1-score" in metrics
    assert all(np.isfinite(v) for v in metrics.values())
