"""The QC cut is a declared policy, so pretraining can loosen it without TS3 moving.

``UnitQCPolicy`` is what a consumer keeps on top of the build, as opposed to
``UnitFiltering`` which is what the build already did. ``core`` ships the mechanism and no
policy: every consumer declares its own beside the dataset that uses it.

What that costs, and these tests buy back: TS3's policy has to stay the one every NEMO and
NuCLR checkpoint was trained under, or new runs are incomparable to the ones in ``ckpt/``,
and Nemo's cache is written under its policy but read under TS3's. The mechanism tests use
a local policy, since they exercise ``UnitQCPolicy`` rather than anyone's choice of one.
"""

import ast
import re
from dataclasses import replace
from importlib import import_module
from inspect import signature
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from core.data import read_recording_ids
from core.dataset import QCNeural, UnitQCPolicy, WholeSessionSpikeDataset
from ts3.ts3_dataset import TS3_UNIT_QC, IBLBrainWideBenchTS3

SRC = Path(__file__).resolve().parents[1]


def _discover_pretrain_policies() -> dict:
    """Public ``*_UNIT_QC`` module constants under ``pretrain/``, by qualified name."""
    found = {}
    for path in sorted((SRC / "pretrain").rglob("*.py")):
        module = None
        for node in ast.walk(ast.parse(path.read_text())):
            if not isinstance(node, ast.Assign):
                continue
            for target in node.targets:
                if isinstance(target, ast.Name) and re.fullmatch(
                    r"[A-Z][A-Z0-9_]*_UNIT_QC", target.id
                ):
                    module = module or import_module(
                        path.relative_to(SRC).with_suffix("").as_posix().replace("/", ".")
                    )
                    found[f"{module.__name__}.{target.id}"] = getattr(module, target.id)
    return found


# Every construction of a whole-session dataset, and every subclass forwarding to one:
# the ``unit_qc`` default means an omission is legal, so the sites are checked instead.
WHOLE_SESSION_CLASSES = {"WholeSessionSpikeDataset"}


def _unit_qc_call_sites(path: Path) -> list[tuple[int, bool]]:
    """Per construction or ``super().__init__`` in a subclass, whether it names a policy."""
    tree = ast.parse(path.read_text())
    subclass = any(
        isinstance(n, ast.ClassDef)
        and any(isinstance(b, ast.Name) and b.id in WHOLE_SESSION_CLASSES for b in n.bases)
        for n in ast.walk(tree)
    )
    sites = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        direct = isinstance(func, ast.Name) and func.id in WHOLE_SESSION_CLASSES
        forwards = (
            subclass
            and isinstance(func, ast.Attribute)
            and func.attr == "__init__"
            and isinstance(func.value, ast.Call)
            and isinstance(func.value.func, ast.Name)
            and func.value.func.id == "super"
        )
        if direct or forwards:
            # ``**kwargs`` does not count: a subclass that forwards everything is the
            # case this is here to catch, not an exemption from it. Nor does an explicit
            # None, which is the parameter's own way of asking for the raw population.
            named = any(
                kw.arg == "unit_qc"
                and not (isinstance(kw.value, ast.Constant) and kw.value.value is None)
                for kw in node.keywords
            )
            sites.append((node.lineno, named))
    return sites


# Only under pretrain/: omitting the policy means the raw population, which an eval-side
# extractor may well want, but which no model should train on by accident.
WHOLE_SESSION_SITES = sorted(
    p.relative_to(SRC).as_posix()
    for p in (SRC / "pretrain").rglob("*.py")
    if _unit_qc_call_sites(p)
)

PROCESSED = Path(__file__).parents[2] / "processed"
PIPELINE_DATA = Path(__file__).parents[2] / "ibl_brain_wide_bench_2026" / "data"

# Exercising the mechanism needs some policy; this one is nobody's.
KEEP_PASSING = UnitQCPolicy(keep_qc_neural=("PASS", "WARNING"))

# Every population a pretraining model trains on, found rather than listed. Public by
# convention: a monitor's own probe cut is private and is not one of these.
PRETRAIN_POLICIES = sorted(_discover_pretrain_policies().items())

requires_local_build = pytest.mark.skipif(
    not (PROCESSED / "ibl_brain_wide_bench_2026" / "pretrain").is_dir(),
    reason="no local processed build to read",
)


def test_ts3_scores_from_the_population_the_checkpoints_saw():
    """FAIL probes out, WARNING in, no rate or label floor, and TS3 cannot be given another."""
    assert TS3_UNIT_QC.keep_qc_neural == ("PASS", "WARNING")
    assert TS3_UNIT_QC.min_firing_rate is None and TS3_UNIT_QC.min_label is None
    # alignment is a label-side judgement, so this policy leaves it alone: a probe whose
    # histology failed still has usable spikes, and every checkpoint saw those units
    assert set(TS3_UNIT_QC.keep_qc_neural_alignment) == set(QCNeural.__args__)
    assert "unit_qc" not in signature(IBLBrainWideBenchTS3).parameters, (
        "TS3 pins its population; a parameter would let a consumer score another one"
    )


@pytest.mark.parametrize("name,policy", PRETRAIN_POLICIES, ids=[n for n, _ in PRETRAIN_POLICIES])
def test_every_pretrainer_trains_on_what_ts3_scores(name, policy):
    """A model may loosen its cut on purpose, but not by leaving a field unset.

    Every field defaults to cutting nothing, so an under-specified policy is silently the
    raw population. Nemo also writes its cache under its own policy while TS3's extractor
    reads that cache under TS3's, so a divergence there rebuilds the whole cache.
    """
    assert policy == TS3_UNIT_QC


@pytest.mark.parametrize("path", WHOLE_SESSION_SITES, ids=lambda p: p)
def test_every_whole_session_dataset_names_its_policy(path):
    """A model cannot inherit a QC policy by accident, only choose one.

    ``unit_qc`` defaults to the raw population, so Python no longer refuses an omission
    and this scan is what does, for the pretraining side where FAIL probes would be
    trained on. Weaker than the required argument it replaces: a construction it cannot
    see statically would slip through.
    """
    for lineno, named in _unit_qc_call_sites(SRC / path):
        assert named, f"{path}:{lineno} builds a whole-session dataset without a unit_qc"


@requires_local_build
def test_a_policy_that_cuts_nothing_covers_the_whole_pretrain_list():
    keep_all = UnitQCPolicy()
    ds = WholeSessionSpikeDataset(root=str(PROCESSED), regime="pretrain", unit_qc=keep_all)
    assert ds.recording_ids == read_recording_ids("pretrain")


@requires_local_build
@pytest.mark.skipif(
    not (PIPELINE_DATA / "bwm_qc.csv").is_file(), reason="pipeline package not checked out"
)
def test_the_sessions_qc_leaves_match_the_pipeline_qc_table():
    """The dataset scans the build for this, so the pipeline's table is what keeps it honest.

    No list of QC-passing sessions is vendored any more: ``bwm_qc.csv`` is where the pipeline
    recorded the same judgement, per probe, and the two must agree.
    """
    qc = pd.read_csv(PIPELINE_DATA / "bwm_qc.csv")
    qc = qc[np.isin(qc.eid, read_recording_ids("pretrain"))]
    passing = qc.groupby("eid").qc_neural.apply(lambda x: (x != "FAIL").any())

    ds = WholeSessionSpikeDataset(root=str(PROCESSED), regime="pretrain", unit_qc=KEEP_PASSING)
    assert ds.recording_ids == passing[passing].index.tolist()


@requires_local_build
def test_the_unit_cut_follows_the_policy():
    """qc_neural is probe level, so a session carries one value per probe, not per unit."""
    ds = WholeSessionSpikeDataset(root=str(PROCESSED), regime="pretrain", unit_qc=KEEP_PASSING)
    rid = next(
        (
            r
            for r in ds.recording_ids[:8]
            if "FAIL" in np.unique(ds.get_recording(r).units.qc_neural.astype(str))
        ),
        None,
    )
    if rid is None:
        pytest.skip("no FAIL units in the first sessions of this build")

    def kept(policy):
        one = WholeSessionSpikeDataset(
            root=str(PROCESSED), regime="pretrain", recording_ids=rid, unit_qc=policy
        )
        return one.dataset_transform(one.get_recording(rid)).units

    shipped = np.unique(ds.get_recording(rid).units.qc_neural.astype(str))
    keep_all = UnitQCPolicy()
    default, unfiltered = kept(KEEP_PASSING), kept(keep_all)
    rate_floor = kept(replace(KEEP_PASSING, min_firing_rate=5.0))

    assert set(np.unique(unfiltered.qc_neural.astype(str))) == set(shipped)
    assert "FAIL" not in np.unique(default.qc_neural.astype(str))
    assert len(default.id) < len(unfiltered.id)
    assert 0 < len(rate_floor.id) < len(default.id)
    assert rate_floor.firing_rate.min() >= 5.0


@requires_local_build
def test_a_named_session_the_policy_empties_is_refused():
    """Honouring part of a request is worse than refusing it, so a named dead session errors.

    The all-FAIL sessions are the case the vendored list used to catch. The all-WARNING one
    under ``keep=("PASS",)`` is the case it could not, since the list only knows FAIL.
    """
    ds = WholeSessionSpikeDataset(root=str(PROCESSED), regime="pretrain", unit_qc=KEEP_PASSING)
    all_fail = sorted(set(read_recording_ids("pretrain")) - set(ds.recording_ids))
    assert all_fail, "QC drops no session in this build, this test is moot"

    with pytest.raises(ValueError, match="leaves no unit"):
        WholeSessionSpikeDataset(
            root=str(PROCESSED), regime="pretrain", unit_qc=KEEP_PASSING, recording_ids=all_fail[:1]
        )

    warning_only = next(
        (
            r
            for r in ds.recording_ids[:8]
            if set(np.unique(ds.get_recording(r).units.qc_neural.astype(str)).tolist())
            == {"WARNING"}
        ),
        None,
    )
    if warning_only is None:
        pytest.skip("this build has no all-WARNING session in the first few")

    with pytest.raises(ValueError, match="leaves no unit"):
        WholeSessionSpikeDataset(
            root=str(PROCESSED),
            regime="pretrain",
            unit_qc=replace(KEEP_PASSING, keep_qc_neural=("PASS",)),
            recording_ids=warning_only,
        )


@requires_local_build
def test_a_policy_that_empties_everything_is_refused():
    with pytest.raises(ValueError, match="leaves no unit"):
        WholeSessionSpikeDataset(
            root=str(PROCESSED),
            regime="pretrain",
            unit_qc=replace(KEEP_PASSING, min_firing_rate=1e6),
            recording_ids=read_recording_ids("pretrain")[0],
        )
