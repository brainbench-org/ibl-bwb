"""The vendored recording-id lists must still agree with the pipeline that produced them.

The lists under ``src/core/data/`` and ``src/ibl_bwb_eval/data/`` are copies of what
``ibl_brain_wide_bench_2026/data/`` generated, so they can silently drift from it. These
tests are the only thing stopping that.
"""

from pathlib import Path

import pytest

from core.data import read_recording_ids

PIPELINE_DATA = Path(__file__).parents[2] / "ibl_brain_wide_bench_2026" / "data"

pytestmark = pytest.mark.skipif(
    not (PIPELINE_DATA / "bwm_qc.csv").is_file(), reason="pipeline package not checked out"
)


@pytest.mark.parametrize("regime", ["pretrain", "eval"])
def test_the_vendored_regime_list_matches_the_pipeline(regime):
    eids = (PIPELINE_DATA / f"{regime}_eids.txt").read_text()
    pipeline_ids = [line.strip() for line in eids.split()]
    assert read_recording_ids(regime) == pipeline_ids
