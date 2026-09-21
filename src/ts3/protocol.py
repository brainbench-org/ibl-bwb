"""How TS3 is evaluated: which units are scored, and the rules applied to predictions.

The unit table here is the ground truth every TS3 metric is computed against. The rest
of the contract lives in ``ibl_bwb_eval``: ``protocol.py`` for the constants shared across all
three suites, ``tasks/ts3.py`` for the region vocabulary, ``multi_unit.py`` for the
multi-unit readout, and
``scoring/ts3.py`` for scoring a submission after the fact.
"""

from pathlib import Path

import numpy as np
import pandas as pd

from core.dataset import BenchmarkRegime, IBLBrainWideBench2026
from core.utils.logger import get_cli_logger
from ibl_bwb_eval.tasks import TS3Task, get_ts3_readout_spec

logger = get_cli_logger()

# Expected neurons after unit and probe-level QC. Per task: a finer atlas resolves fewer
# units, so the sentinel cut at the end of the cascade removes a different number.
_EXPECTED_MD_SIZE: dict[str, dict[str, int]] = {
    "unit_cosmos": {"pretrain": 41_834, "eval": 3_222},
}


def get_md_from_dataset(data_root: Path, regime: BenchmarkRegime, task: TS3Task) -> pd.DataFrame:
    """The unit table for one regime, after the QC every TS3 metric is computed against.

    The cascade below is load-time QC, and the two regimes do not share a rule: eval keeps
    ``qc_neural == PASS`` while pretrain keeps everything but ``FAIL``, so units on
    ``WARNING`` probes are trained on and never scored. The guide's :ref:`ts3_unit_qc` has
    the full table, and ``_EXPECTED_MD_SIZE`` pins the count a rebuild must reproduce.
    """
    spec = get_ts3_readout_spec(task)

    ds = IBLBrainWideBench2026(
        root=str(data_root),
        split=None,
        regime=regime,
        require_unit_filtering="selected_units" if regime == "eval" else None,
        contract="TS3 scoring",
    )

    unit_ids = []
    regions = []
    session_ids = []
    subject_ids = []
    depths = []
    probe_ids = []
    qc_neural_alignments = []
    qc_neurals = []
    firing_rates = []
    unit_qc_labels = []

    for rid in ds.recording_ids:
        rec = ds.get_recording(rid)
        unit_ids.append(rec.units.id.astype(str))
        regions.append(getattr(rec.units, spec.region_key).astype(str))
        session_ids.extend([rid] * len(rec.units))
        subject_ids.extend([rec.subject.id] * len(rec.units))
        depths.extend(rec.units.depths)
        probe_ids.extend(rec.units.probe_id.astype(str))
        qc_neural_alignments.extend(rec.units.qc_neural_alignment.astype(str))
        qc_neurals.extend(rec.units.qc_neural.astype(str))
        firing_rates.extend(rec.units.firing_rate)
        unit_qc_labels.extend(rec.units.label)

    ans = pd.DataFrame(
        {
            "unit_id": np.concatenate(unit_ids),
            "brain_region": np.concatenate(regions),
            "session_id": session_ids,
            "subject_id": subject_ids,
            "depths": depths,
            "probe_ids": probe_ids,
            "qc_neural_alignment": qc_neural_alignments,
            "qc_neural": qc_neurals,
            "firing_rate": firing_rates,
            "unit_qc_label": unit_qc_labels,
        }
    ).set_index("unit_id")
    # The scoring cut, stricter than the dataset's: the alignment, region and pretrain
    # WARNING lines are load-time only; the label and rate floors re-verify the build.
    if regime == "eval":
        ans = ans[ans.qc_neural.values == "PASS"]
    else:
        ans = ans[ans.qc_neural.values != "FAIL"]
    ans = ans[ans.qc_neural_alignment.values == "PASS"]
    ans = ans[ans.unit_qc_label == 1.0]
    ans = ans[ans.firing_rate > 1.0]
    ans = ans[~np.isin(ans.brain_region.values, ["void", "root"])]

    exp_size = _EXPECTED_MD_SIZE[task][regime]
    if len(ans) != exp_size:
        raise RuntimeError(
            f"Number of candidate {task} {regime} units doesn't match expectation. "
            f"Expected: {exp_size}, Got: {len(ans)}"
        )

    ans.attrs = {"unit_filtering": ds.unit_filtering.label, "dataset_version": ds.dataset_version}

    return ans
