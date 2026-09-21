"""How the v01 legacy prediction fixtures in this directory were written.

v01 is the shape ``PredictionsWriter`` produced before it stamped its own version.
These go
through ``save_file`` rather than the writer, so a new metadata field can never move the
committed bytes: one file per suite, ts1-choice logits, ts2-co_smoothing rates and
ts3-unit_cosmos probabilities.

Run from the repo root: ``python src/tests/fixtures/legacy/make_v01_fixtures.py``
"""

from pathlib import Path

import numpy as np
import torch
from safetensors.torch import save_file

from ibl_bwb_eval.entity_ids import encode_entity_ids
from ibl_bwb_eval.tasks import COSMOS_LABELS

HERE = Path(__file__).resolve().parent
RECORDING_ID = "5ec72172-3901-4771-8777-6e9490ca51fc"


def _unit_ids(n: int) -> torch.Tensor:
    session = "a" * 36
    return encode_entity_ids(np.array([f"{session}/{i:036d}" for i in range(n)]))


def make_ts1() -> None:
    g = torch.Generator().manual_seed(1)
    n = 16
    save_file(
        {
            "predictions": torch.randn(n, 1, 2, generator=g),  # (N, 1, D=2) logits
            "trial_id": torch.arange(n, dtype=torch.int64),
            "timestamps": torch.zeros(n, 1),
        },
        str(HERE / "v01_ts01.dat"),
        metadata={
            "label": "legacy-v01",
            "task": "ts1-choice",
            "seed": "42",
            "recording_id": RECORDING_ID,
        },
    )


def make_ts2() -> None:
    g = torch.Generator().manual_seed(2)
    w, t, u = 8, 10, 4
    save_file(
        {
            "predictions": torch.rand(w, t, u, generator=g),
            "window_timestamps": torch.arange(w, dtype=torch.float32),
            "unit_ids": _unit_ids(u),
        },
        str(HERE / "v01_ts02.dat"),
        metadata={
            "label": "legacy-v01",
            "task": "ts2-co_smoothing",
            "seed": "42",
            "recording_id": RECORDING_ID,
        },
    )


def make_ts3() -> None:
    g = torch.Generator().manual_seed(3)
    n, c = 20, len(COSMOS_LABELS)
    save_file(
        {
            "pred_proba": torch.rand(n, c, generator=g),
            "entity_ids": _unit_ids(n),
        },
        str(HERE / "v01_ts03.dat"),
        metadata={
            "label": "legacy-v01",
            "task": "ts3-unit_cosmos",
            "seed": "42",
            "label_names": ",".join(COSMOS_LABELS),
        },
    )


if __name__ == "__main__":
    make_ts1()
    make_ts2()
    make_ts3()
    print(f"Wrote v01 fixtures to {HERE}")
