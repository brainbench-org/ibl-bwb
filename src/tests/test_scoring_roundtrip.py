"""Format round-trip tests for the scoring read path (ts1/ts2/ts3).

These write tiny, synthetic prediction files with the real :class:`PredictionsWriter`
and matching ground-truth ``.safetensors`` files, then score them through
``ibl_bwb_eval.scoring.<suite>.score_dir``. They exercise the on-disk format and alignment
logic end-to-end without any committed dataset, so they stay in lock-step with the
format as it evolves (change the writer/reader -> these break in the same PR).

Metric *values* are checked only for finiteness and key presence; numerical
correctness of the metrics themselves lives in ``test_bps`` / ``test_poisson_d2``.
"""

from pathlib import Path

import numpy as np
import pytest
import torch
from safetensors.torch import save_file

from ibl_bwb_eval.entity_ids import encode_entity_ids
from ibl_bwb_eval.predictions import PredictionsWriter
from ibl_bwb_eval.scoring import ts1 as ts1_scoring
from ibl_bwb_eval.scoring import ts2 as ts2_scoring
from ibl_bwb_eval.scoring import ts3 as ts3_scoring
from ibl_bwb_eval.tasks import COSMOS_LABELS

RECORDING_ID = "5ec72172-3901-4771-8777-6e9490ca51fc"
TS3_LABELS = ",".join(COSMOS_LABELS)


def _unit_ids(n: int) -> torch.Tensor:
    """n synthetic 73-char '{session}/{unit}' ids, encoded as the on-disk uint8 tensor."""
    session = "a" * 36
    return encode_entity_ids(np.array([f"{session}/{i:036d}" for i in range(n)]))


def _write_pred(base: Path, task: str, seed: int, fields: dict, metadata: dict) -> None:
    """Save a prediction file via the real writer (the code that produces submissions)."""
    writer = PredictionsWriter(
        enable=True, base_path=base, task=task, seed=seed, label="synthetic", metadata=metadata
    )
    writer.set(**fields)
    writer.save()


def test_ts1_roundtrip(tmp_path):
    """A sequence-level TS1 classification file scores to the readout-spec metrics."""
    pred_dir, gt_dir = tmp_path.joinpath("pred"), tmp_path.joinpath("gt")
    n = 16
    trial_id = torch.arange(n, dtype=torch.int64)

    _write_pred(
        pred_dir,
        task="ts1-choice",
        seed=42,
        fields={
            "predictions": torch.randn(n, 1, 2),  # (N, 1, D=2) logits
            "trial_id": trial_id,
            "timestamps": torch.zeros(n, 1),
        },
        metadata={"recording_id": RECORDING_ID},
    )

    gt_path = gt_dir.joinpath("ts1-choice", RECORDING_ID, "ground_truth.safetensors")
    gt_path.parent.mkdir(parents=True, exist_ok=True)
    save_file(
        {
            "values": torch.randint(0, 2, (n, 1)).float(),
            "trial_id": trial_id,
            "timestamps": torch.zeros(n, 1),
        },
        str(gt_path),
        metadata={"task": "ts1-choice", "recording_id": RECORDING_ID},
    )

    raw = ts1_scoring.score_dir(pred_dir, gt_dir)
    assert ("synthetic", "ts1-choice", RECORDING_ID, 42) in raw
    metrics = raw[("synthetic", "ts1-choice", RECORDING_ID, 42)]
    assert {"bacc", "f1", "ap"} <= set(metrics)
    assert all(np.isfinite(v) for v in metrics.values())


def _write_ts1_paw_case(tmp_path, mask, *, write_mask: bool = True):
    """A masked timestep-level TS1 case, with garbage wherever ``mask`` is False.

    Masked-out slots hold values no model could produce, so any regression that stops
    applying the mask moves r2 by whole units instead of decimals.
    """
    pred_dir, gt_dir = tmp_path.joinpath("pred"), tmp_path.joinpath("gt")
    n, t, d = 8, 20, 1
    g = torch.Generator().manual_seed(0)
    values = torch.randn(n, t, d, generator=g)
    predictions = values + 0.05 * torch.randn(n, t, d, generator=g)
    bad = ~mask.bool()
    predictions[bad], values[bad] = 99.0, -99.0

    trial_id = torch.arange(n, dtype=torch.int64)
    _write_pred(
        pred_dir,
        task="ts1-right_paw_speed",
        seed=42,
        fields={
            "predictions": predictions,
            "trial_id": trial_id,
            "timestamps": torch.zeros(n, t),
        },
        metadata={"recording_id": RECORDING_ID},
    )

    gt_path = gt_dir.joinpath("ts1-right_paw_speed", RECORDING_ID, "ground_truth.safetensors")
    gt_path.parent.mkdir(parents=True, exist_ok=True)
    gt = {"values": values, "trial_id": trial_id, "timestamps": torch.zeros(n, t)}
    if write_mask:
        gt["mask"] = mask
    save_file(
        gt,
        str(gt_path),
        metadata={"task": "ts1-right_paw_speed", "recording_id": RECORDING_ID},
    )
    return next(pred_dir.rglob("seed_*.safetensors")), gt_path, pred_dir, gt_dir


def test_ts1_masked_roundtrip(tmp_path):
    """A masked TS1 task scores only the timesteps the ground-truth mask keeps."""
    mask = torch.rand(8, 20, generator=torch.Generator().manual_seed(1)) > 0.3
    _, _, pred_dir, gt_dir = _write_ts1_paw_case(tmp_path, mask)

    raw = ts1_scoring.score_dir(pred_dir, gt_dir)
    metrics = raw[("synthetic", "ts1-right_paw_speed", RECORDING_ID, 42)]

    assert {"r2", "mae", "pearson"} <= set(metrics)
    assert all(np.isfinite(v) for v in metrics.values())
    # only reachable if every masked-out slot was dropped before the metrics saw it
    assert metrics["r2"] > 0.9
    assert metrics["mae"] < 1.0


def test_ts1_missing_mask_is_rejected(tmp_path):
    """A task declaring mask_key must not silently fall back to unmasked scoring."""
    mask = torch.rand(8, 20, generator=torch.Generator().manual_seed(1)) > 0.3
    pred_path, gt_path, _, _ = _write_ts1_paw_case(tmp_path, mask, write_mask=False)

    with pytest.raises(ValueError, match="carries no"):
        ts1_scoring.score_file("ts1-right_paw_speed", pred_path, gt_path)


def test_ts1_empty_mask_is_rejected(tmp_path):
    """A mask that keeps nothing is an error, not a zero-length metric update."""
    mask = torch.zeros(8, 20, dtype=torch.bool)
    pred_path, gt_path, _, _ = _write_ts1_paw_case(tmp_path, mask)

    with pytest.raises(ValueError, match="no unmasked timestep"):
        ts1_scoring.score_file("ts1-right_paw_speed", pred_path, gt_path)


def test_ts2_roundtrip(tmp_path):
    """A TS2 file aligns to GT by window_timestamps and scores poisson_d2 + bps."""
    pred_dir, gt_dir = tmp_path.joinpath("pred"), tmp_path.joinpath("gt")
    w, t, u = 32, 50, 6  # windows, timesteps, units
    window_timestamps = torch.arange(w, dtype=torch.float32)

    _write_pred(
        pred_dir,
        task="ts2-co_smoothing",
        seed=42,
        fields={
            "predictions": torch.rand(w, t, u),
            "window_timestamps": window_timestamps,
            "unit_ids": _unit_ids(u),
        },
        metadata={"recording_id": RECORDING_ID},
    )

    gt_path = gt_dir.joinpath("ts2-co_smoothing", RECORDING_ID, "ground_truth.safetensors")
    gt_path.parent.mkdir(parents=True, exist_ok=True)
    save_file(
        {
            "targets": torch.randint(0, 5, (w, t, u)).float(),
            "window_timestamps": window_timestamps,
            "unit_ids": _unit_ids(u),
        },
        str(gt_path),
        metadata={"task": "ts2-co_smoothing", "recording_id": RECORDING_ID},
    )

    raw = ts2_scoring.score_dir(pred_dir, gt_dir)
    metrics = raw[("synthetic", "ts2-co_smoothing", RECORDING_ID, 42)]
    assert {"poisson_d2", "bps"} <= set(metrics)
    assert all(np.isfinite(v) for v in metrics.values())


def test_ts3_roundtrip(tmp_path):
    """A TS3 file aligns to GT by unit_id and scores macro/per-region classification."""
    pred_dir, gt_dir = tmp_path.joinpath("pred"), tmp_path.joinpath("gt")
    n, c = 40, TS3_LABELS.count(",") + 1  # units, classes
    entity_ids = _unit_ids(n)

    _write_pred(
        pred_dir,
        task="ts3-unit_cosmos",
        seed=42,
        fields={"pred_proba": torch.rand(n, c), "entity_ids": entity_ids},
        metadata={"label_names": TS3_LABELS},
    )

    gt_path = gt_dir.joinpath("ts3-unit_cosmos", "ground_truth.safetensors")
    gt_path.parent.mkdir(parents=True, exist_ok=True)
    save_file(
        {"targets": torch.randint(0, c, (n,)), "entity_ids": entity_ids},
        str(gt_path),
        metadata={"label_names": TS3_LABELS},
    )

    raw = ts3_scoring.score_dir(pred_dir, gt_dir)
    metrics = raw[("synthetic", 42)]
    assert "macro/f1-score" in metrics
    assert all(np.isfinite(v) for v in metrics.values())


@pytest.mark.parametrize("mod", [ts1_scoring, ts2_scoring, ts3_scoring])
def test_score_dir_empty(mod, tmp_path):
    """An empty prediction tree returns an empty result rather than raising."""
    assert mod.score_dir(tmp_path, tmp_path) == {}
