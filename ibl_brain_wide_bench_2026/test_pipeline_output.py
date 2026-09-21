"""CI integration test for the pipeline output.

Runs the pipeline on two eval sessions, validates the output h5 against hardcoded
expected values and the published S3 references, then removes the temp files.

Run with:
    cd ibl_brain_wide_bench_2026
    python -m pytest test_pipeline_output.py -v

Skipped automatically when the IBL pipeline dependencies (ONE-api, ibllib, …) are
missing.  Set IBL_RAW_DIR and S3_H5_CACHE_DIR to persistent directories to avoid
re-downloading on every run.
"""

import os
import sys
import urllib.request
from dataclasses import dataclass, field
from itertools import pairwise
from pathlib import Path

import h5py
import numpy as np
import pytest

one = pytest.importorskip("one", reason="IBL pipeline deps not installed")

sys.path.insert(0, str(Path(__file__).parent))
import pipeline as _pipeline  # noqa: E402  (needs sys.path insert above)

# ── target session ────────────────────────────────────────────────────────────

# Two probes, one PASS and one FAIL, so cross-probe concatenation and per-probe QC
# are both exercised.
EID = "16693458-0801-4d35-a3f1-9115c7e5acfd"

# Its published S3 file was built probe01-first, so comparisons against it must be
# order-agnostic.
EID_REORDERED = "0cbeae00-e229-4b7d-bdcc-1b0569d7e0c3"


def _s3_url(eid: str) -> str:
    return (
        "https://brain-wide-bench.s3.amazonaws.com/brainsets/all_units/"
        f"ibl_brain_wide_bench_2026/eval/{eid}.h5"
    )


# ── hardcoded expected values ─────────────────────────────────────────────────

SPLIT_GAPS_MIN_S = 20.0

EXPECTED_TRAIN_DUR_S = 1_139.1346
EXPECTED_VAL_DUR_S = 494.7684
EXPECTED_TEST_DUR_S = 1_242.0985

EXPECTED_TS2_TRAIN_DUR_S = 1_132.9334
EXPECTED_TS2_VAL_DUR_S = 548.8504
EXPECTED_TS2_TEST_DUR_S = 1_034.1423

EXPECTED_N_UNITS = 1_912
EXPECTED_MEAN_FR = 7.925738

# positionally aligned: units are concatenated in probe-name order
EXPECTED_PROBE_NAMES = ["probe00", "probe01"]
EXPECTED_PROBE_IDS = [
    "53ecbf4f-e0d8-4fe6-a852-8b934a37a1c2",
    "b543e81e-4c8f-415e-82ec-631b177d19d2",
]
EXPECTED_UNITS_PER_PROBE = [917, 995]
EXPECTED_PROBE_QC_NEURAL = ["PASS", "FAIL"]

EXPECTED_BEHAVIOR_SFREQ = 50

EXPECTED_WHISKER_MEAN = 5.384172
EXPECTED_WHISKER_STD = 4.201384
EXPECTED_WHEEL_MEAN = 0.276059
EXPECTED_WHEEL_STD = 0.613214
EXPECTED_LICK_MEAN = 8.259788
EXPECTED_LICK_STD = 33.876974

# ts1_normalize/ must contain exactly these keys (no wheel, no licks)
EXPECTED_NORMALIZE_KEYS = {"paws", "whisker"}

EXPECTED_NORM_WHISKER_MEAN = 9.3165763
EXPECTED_NORM_WHISKER_STD = 2.4442013
EXPECTED_NORM_LEFT_PAW_SPEED_MEAN = 609.53110044
EXPECTED_NORM_LEFT_PAW_SPEED_STD = 671.67642129
EXPECTED_NORM_RIGHT_PAW_SPEED_MEAN = 426.80779456
EXPECTED_NORM_RIGHT_PAW_SPEED_STD = 489.49344819

EXPECTED_N_TRIALS = 531
EXPECTED_REWARD_RATE = 0.828625
EXPECTED_MEAN_RT_S = 2.141672

# ── fixture ───────────────────────────────────────────────────────────────────


@dataclass
class _Args:
    reprocess: bool = True
    list_sessions: str = False
    download_first: bool = True
    unit_filter: tuple = field(default_factory=tuple)
    small: bool = True
    example_frame: bool = False
    api_retries: int = _pipeline.DEFAULT_API_RETRIES


@pytest.fixture(scope="module")
def run_pipeline(tmp_path_factory):
    """Process a session on demand, at most once per eid."""
    raw_dir = Path(os.environ.get("IBL_RAW_DIR", tmp_path_factory.mktemp("raw"))).expanduser()
    out_dir = tmp_path_factory.mktemp("processed")

    args = _Args()
    p = _pipeline.Pipeline(raw_dir=raw_dir, processed_dir=out_dir, args=args)
    manifest = p.get_manifest(raw_dir, args)
    done: dict[str, Path] = {}

    def _run(eid: str) -> Path:
        if eid not in done:
            p.process(manifest.loc[eid])
            path = out_dir / "eval" / f"{eid}.h5"
            assert path.exists(), f"Pipeline produced no output at {path}"
            done[eid] = path
        return done[eid]

    yield _run
    for path in done.values():
        path.unlink(missing_ok=True)


@pytest.fixture(scope="module")
def h5_path(run_pipeline):
    return run_pipeline(EID)


def _fetch_s3(eid: str, tmp_path_factory) -> Path:
    cache_dir = os.environ.get("S3_H5_CACHE_DIR")
    if cache_dir:
        path = Path(cache_dir) / f"{eid}.h5"
        path.parent.mkdir(parents=True, exist_ok=True)
    else:
        path = tmp_path_factory.mktemp("s3") / f"{eid}.h5"
    if not path.exists():
        urllib.request.urlretrieve(_s3_url(eid), path)
    return path


@pytest.fixture(scope="module", params=[EID, EID_REORDERED], ids=["primary", "reordered"])
def session(request, run_pipeline, tmp_path_factory):
    """(local output, S3 reference) for each session, including a reordered one."""
    eid = request.param
    return run_pipeline(eid), _fetch_s3(eid, tmp_path_factory)


# ── helpers ───────────────────────────────────────────────────────────────────


def _interval_duration(f, key):
    starts = f[f"{key}/start"][:]
    ends = f[f"{key}/end"][:]
    return float((ends - starts).sum())


def _strings(f, key):
    return [s.decode() if isinstance(s, bytes) else s for s in f[key][:]]


# ── tests ─────────────────────────────────────────────────────────────────────


def test_split_causal_order(h5_path):
    with h5py.File(h5_path) as f:
        train_end = float(f["ts1_train_domain/end"][:][-1])
        val_start = float(f["ts1_val_domain/start"][:][0])
        val_end = float(f["ts1_val_domain/end"][:][-1])
        test_start = float(f["ts1_test_domain/start"][:][0])

    assert train_end < val_start, "val must start after train ends"
    assert val_end < test_start, "test must start after val ends"


def test_split_gaps(h5_path):
    with h5py.File(h5_path) as f:
        train_end = float(f["ts1_train_domain/end"][:][-1])
        val_start = float(f["ts1_val_domain/start"][:][0])
        val_end = float(f["ts1_val_domain/end"][:][-1])
        test_start = float(f["ts1_test_domain/start"][:][0])

    assert val_start - train_end >= SPLIT_GAPS_MIN_S
    assert test_start - val_end >= SPLIT_GAPS_MIN_S


def test_ts2_split_gaps(h5_path):
    # no same-split neighbours in the pattern, so this walks every boundary
    with h5py.File(h5_path) as f:
        chunks = sorted(
            (float(a), float(b), split)
            for split in ("train", "val", "test")
            for a, b in zip(
                f[f"ts2_{split}_domain/start"][:], f[f"ts2_{split}_domain/end"][:], strict=True
            )
        )

    assert len(chunks) > 1, "expected more than one chunk"
    for (_, left_end, left), (right_start, _, right) in pairwise(chunks):
        if left != right:
            assert right_start - left_end >= SPLIT_GAPS_MIN_S, (
                f"{left}->{right} boundary is only {right_start - left_end:.2f}s wide"
            )


def test_ts2_split_durations(h5_path):
    with h5py.File(h5_path) as f:
        train_dur = _interval_duration(f, "ts2_train_domain")
        val_dur = _interval_duration(f, "ts2_val_domain")
        test_dur = _interval_duration(f, "ts2_test_domain")

    assert train_dur == pytest.approx(EXPECTED_TS2_TRAIN_DUR_S, abs=0.1)
    assert val_dur == pytest.approx(EXPECTED_TS2_VAL_DUR_S, abs=0.1)
    assert test_dur == pytest.approx(EXPECTED_TS2_TEST_DUR_S, abs=0.1)


def test_split_durations(h5_path):
    with h5py.File(h5_path) as f:
        train_dur = _interval_duration(f, "ts1_train_domain")
        val_dur = _interval_duration(f, "ts1_val_domain")
        test_dur = _interval_duration(f, "ts1_test_domain")

    assert train_dur == pytest.approx(EXPECTED_TRAIN_DUR_S, abs=0.1)
    assert val_dur == pytest.approx(EXPECTED_VAL_DUR_S, abs=0.1)
    assert test_dur == pytest.approx(EXPECTED_TEST_DUR_S, abs=0.1)


def test_unit_filtering_metadata(h5_path):
    # this suite builds unfiltered, and src/core/data/unit_filtering.py reads these back
    with h5py.File(h5_path) as f:
        attrs = f["brainset"].attrs
        assert attrs["unit_filtering"] == "all_units"
        assert attrs["unit_filters"] == ""
        assert attrs["unit_filter_thresholds"] == ""


def test_n_units(h5_path):
    with h5py.File(h5_path) as f:
        n = len(f["units/firing_rate"][:])
    assert n == EXPECTED_N_UNITS


# ── multi-probe concatenation ─────────────────────────────────────────────────
def test_probes_present_and_ordered(h5_path):
    with h5py.File(h5_path) as f:
        assert _strings(f, "probes/name") == EXPECTED_PROBE_NAMES
        assert _strings(f, "probes/id") == EXPECTED_PROBE_IDS


def test_per_probe_unit_counts(h5_path):
    with h5py.File(h5_path) as f:
        probe_ids = _strings(f, "probes/id")
        unit_probe = np.asarray(_strings(f, "units/probe_id"))

    counts = [int((unit_probe == pid).sum()) for pid in probe_ids]
    assert counts == EXPECTED_UNITS_PER_PROBE
    assert sum(counts) == EXPECTED_N_UNITS, "every unit must belong to a listed probe"


def test_probe_and_unit_order_is_name_sorted(session):
    """Probe order must come from the data, not from ONE's return order.

    ``eid2pid`` sorts by name only on its cache path; on the remote path it passes REST
    order through. Asserting *sorted* rather than self-consistent is the point -- both
    orders are internally consistent, so only a fixed one catches the drift.
    """
    local, _ = session
    with h5py.File(local) as f:
        names = _strings(f, "probes/name")
        ids = _strings(f, "probes/id")
        unit_probe = _strings(f, "units/probe_id")

    assert names == sorted(names), f"probes table is not name-sorted: {names}"
    id_to_name = dict(zip(ids, names, strict=True))
    block_names = [id_to_name[p] for p in dict.fromkeys(unit_probe)]
    assert block_names == sorted(block_names), f"unit blocks are not name-sorted: {block_names}"


def test_units_are_contiguous_per_probe(h5_path):
    """Each probe owns one contiguous range; a unit_offset regression would interleave them."""
    with h5py.File(h5_path) as f:
        probe_ids = _strings(f, "probes/id")
        unit_probe = np.asarray(_strings(f, "units/probe_id"))

    boundaries = np.flatnonzero(unit_probe[1:] != unit_probe[:-1]) + 1
    assert len(boundaries) == len(probe_ids) - 1, "probe blocks are not contiguous"
    assert list(dict.fromkeys(unit_probe.tolist())) == probe_ids, "block order != probe order"


def test_spike_unit_indices_span_all_probes(h5_path):
    """Spike unit_index must address the full concatenated unit table, not probe-local ids."""
    with h5py.File(h5_path) as f:
        unit_index = f["spikes/unit_index"][:]
        n_units = len(f["units/id"][:])

    assert unit_index.min() >= 0
    # a missing offset would cap this at the first probe's unit count
    assert unit_index.max() == n_units - 1
    assert len(np.unique(unit_index)) == n_units, "every unit must have spikes"


def test_unit_ids_unique_across_probes(h5_path):
    with h5py.File(h5_path) as f:
        unit_ids = _strings(f, "units/id")
    assert len(set(unit_ids)) == len(unit_ids)


def test_per_probe_qc_propagated(h5_path):
    """The two probes have different neural QC; each unit inherits its own probe's verdict."""
    with h5py.File(h5_path) as f:
        assert _strings(f, "probes/qc_neural") == EXPECTED_PROBE_QC_NEURAL
        probe_ids = _strings(f, "probes/id")
        unit_probe = np.asarray(_strings(f, "units/probe_id"))
        unit_qc = np.asarray(_strings(f, "units/qc_neural"))

    for pid, expected_qc in zip(probe_ids, EXPECTED_PROBE_QC_NEURAL, strict=True):
        vals = set(unit_qc[unit_probe == pid].tolist())
        assert vals == {expected_qc}, f"probe {pid} units have qc_neural={vals}"


def test_mean_firing_rate(h5_path):
    with h5py.File(h5_path) as f:
        spike_unit = f["spikes/unit_index"][:]
        n_units = len(f["units/id"][:])
        duration = float((f["spikes/domain/end"][:] - f["spikes/domain/start"][:]).sum())
        spike_counts = np.bincount(spike_unit, minlength=n_units)
        mean_fr = float(spike_counts.mean()) / duration
    assert mean_fr == pytest.approx(EXPECTED_MEAN_FR, rel=1e-3)


def test_behavioral_sampling_rates(h5_path):
    with h5py.File(h5_path) as f:
        for sig in ["whisker", "wheel", "licks", "paws"]:
            sfreq = int(f[sig].attrs["sampling_rate"])
            assert sfreq == EXPECTED_BEHAVIOR_SFREQ, f"{sig} sfreq={sfreq}"


def test_behavioral_signal_stats(h5_path):
    with h5py.File(h5_path) as f:
        whisker = f["whisker/whisker_motion_energy"][:]
        wheel = f["wheel/wheel_speed"][:]
        lick = f["licks/licking_rate"][:]

    assert np.nanmean(whisker) == pytest.approx(EXPECTED_WHISKER_MEAN, rel=1e-4)
    assert np.nanstd(whisker) == pytest.approx(EXPECTED_WHISKER_STD, rel=1e-4)
    assert np.nanmean(wheel) == pytest.approx(EXPECTED_WHEEL_MEAN, rel=1e-4)
    assert np.nanstd(wheel) == pytest.approx(EXPECTED_WHEEL_STD, rel=1e-4)
    assert np.nanmean(lick) == pytest.approx(EXPECTED_LICK_MEAN, rel=1e-4)
    assert np.nanstd(lick) == pytest.approx(EXPECTED_LICK_STD, rel=1e-4)


def test_no_nans_in_behavioral_signals(h5_path):
    with h5py.File(h5_path) as f:
        for path in [
            "whisker/whisker_motion_energy",
            "wheel/wheel_speed",
            "licks/licking_rate",
        ]:
            arr = f[path][:]
            assert not np.any(np.isnan(arr)), f"NaNs found in {path}"


def test_normalization_scope(h5_path):
    """Wheel and licks must not appear in ts1_normalize/ (only pixel signals are z-scored)."""
    with h5py.File(h5_path) as f:
        keys = set(f["ts1_normalize"].keys())
    assert keys == EXPECTED_NORMALIZE_KEYS, f"ts1_normalize/ keys: {keys}"


def test_normalization_stats(h5_path):
    with h5py.File(h5_path) as f:
        wm = float(f["ts1_normalize/whisker/whisker_motion_energy"].attrs["mean"])
        ws = float(f["ts1_normalize/whisker/whisker_motion_energy"].attrs["std"])
        lm = float(f["ts1_normalize/paws/left_paw_speed"].attrs["mean"])
        ls = float(f["ts1_normalize/paws/left_paw_speed"].attrs["std"])
        rm = float(f["ts1_normalize/paws/right_paw_speed"].attrs["mean"])
        rs = float(f["ts1_normalize/paws/right_paw_speed"].attrs["std"])

    assert wm == pytest.approx(EXPECTED_NORM_WHISKER_MEAN, rel=1e-5)
    assert ws == pytest.approx(EXPECTED_NORM_WHISKER_STD, rel=1e-5)
    assert lm == pytest.approx(EXPECTED_NORM_LEFT_PAW_SPEED_MEAN, rel=1e-5)
    assert ls == pytest.approx(EXPECTED_NORM_LEFT_PAW_SPEED_STD, rel=1e-5)
    assert rm == pytest.approx(EXPECTED_NORM_RIGHT_PAW_SPEED_MEAN, rel=1e-5)
    assert rs == pytest.approx(EXPECTED_NORM_RIGHT_PAW_SPEED_STD, rel=1e-5)


def test_trial_count(h5_path):
    with h5py.File(h5_path) as f:
        n = len(f["trials/start"][:])
    assert n == EXPECTED_N_TRIALS


def test_reward_rate(h5_path):
    with h5py.File(h5_path) as f:
        rate = float(f["trials/reward"][:].mean())
    assert rate == pytest.approx(EXPECTED_REWARD_RATE, abs=1e-4)


def test_mean_rt(h5_path):
    with h5py.File(h5_path) as f:
        rt = f["trials/response_time"][:] - f["trials/stim_on_time"][:]
    assert float(np.nanmean(rt)) == pytest.approx(EXPECTED_MEAN_RT_S, abs=1e-3)


# ── S3 reference comparison ───────────────────────────────────────────────────


@pytest.fixture(scope="module")
def s3_h5_path(tmp_path_factory):
    """Reference HDF5 for the primary session; set S3_H5_CACHE_DIR to persist it."""
    return _fetch_s3(EID, tmp_path_factory)


# The S3 files predate the probe-order fix, so comparisons against them must be keyed by
# unit id, not column index.


def test_s3_unit_population_matches(session):
    local, ref = session
    with h5py.File(local) as loc, h5py.File(ref) as r:
        assert sorted(_strings(loc, "units/id")) == sorted(_strings(r, "units/id"))


def test_s3_spike_to_unit_assignment_matches(session):
    """Every spike keeps its neuron, up to same-timestamp swaps.

    Stream position is not a stable identity: ``spikes.sort()`` does not preserve order
    among equal timestamps, and ~11% of spikes are tied, so a differing probe order
    permutes them without changing any neuron's spike train. Hence the two assertions --
    a spike may only change neuron on an identical timestamp, and the neurons firing at
    that timestamp must be unchanged. With test_s3_spike_timestamps_match pinning the
    timestamps elementwise, that is per-neuron spike-train equality.

    The unit remap runs over the unit table, never per spike: ids are 36-char UUIDs that
    numpy widens to 144-byte <U36, so one per spike costs ~6 GB and OOMs the runner.
    """
    local, ref = session
    with h5py.File(local) as loc, h5py.File(ref) as r:
        pos = {uid: i for i, uid in enumerate(_strings(loc, "units/id"))}
        ref_to_loc = np.array([pos[uid] for uid in _strings(r, "units/id")], dtype=np.int64)
        loc_units = loc["spikes/unit_index"][:]
        ref_units = ref_to_loc[r["spikes/unit_index"][:]]
        moved = np.flatnonzero(loc_units != ref_units)
        # subset immediately; the full timestamp columns are ~0.5 GB each
        loc_times = loc["spikes/timestamps"][:]
        moved_loc_times = loc_times[moved]
        del loc_times
        ref_times = r["spikes/timestamps"][:]
        moved_ref_times = ref_times[moved]
        del ref_times

    assert np.array_equal(moved_loc_times, moved_ref_times), (
        "a spike changed neuron without landing on an identical timestamp"
    )

    moved_loc_units = loc_units[moved]
    moved_ref_units = ref_units[moved]
    loc_order = np.lexsort((moved_loc_units, moved_loc_times))
    ref_order = np.lexsort((moved_ref_units, moved_ref_times))
    assert np.array_equal(moved_loc_units[loc_order], moved_ref_units[ref_order]), (
        "the set of neurons firing at a tied timestamp differs"
    )


def test_s3_spike_timestamps_match(session):
    local, ref = session
    with h5py.File(local) as loc, h5py.File(ref) as r:
        assert np.array_equal(loc["spikes/timestamps"][:], r["spikes/timestamps"][:])


def test_s3_splits_match(h5_path, s3_h5_path):
    with h5py.File(h5_path) as loc, h5py.File(s3_h5_path) as ref:
        for key in [
            "ts1_train_domain/start",
            "ts1_train_domain/end",
            "ts1_val_domain/start",
            "ts1_val_domain/end",
            "ts1_test_domain/start",
            "ts1_test_domain/end",
        ]:
            assert np.array_equal(loc[key][:], ref[key][:]), f"mismatch in {key}"


def test_s3_trials_match(h5_path, s3_h5_path):
    with h5py.File(h5_path) as loc, h5py.File(s3_h5_path) as ref:
        assert np.array_equal(loc["trials/start"][:], ref["trials/start"][:])
        assert np.array_equal(loc["trials/reward"][:], ref["trials/reward"][:])


def test_s3_behavior_match(h5_path, s3_h5_path):
    with h5py.File(h5_path) as loc, h5py.File(s3_h5_path) as ref:
        for path in [
            "whisker/whisker_motion_energy",
            "wheel/wheel_speed",
            "licks/licking_rate",
        ]:
            np.testing.assert_allclose(
                loc[path][:],
                ref[path][:],
                rtol=1e-5,
                atol=1e-10,
                err_msg=f"mismatch in {path}",
            )


def test_s3_normalization_match(h5_path, s3_h5_path):
    with h5py.File(h5_path) as loc, h5py.File(s3_h5_path) as ref:
        for path in [
            "ts1_normalize/whisker/whisker_motion_energy",
            "ts1_normalize/paws/left_paw_speed",
            "ts1_normalize/paws/right_paw_speed",
        ]:
            for attr in ("mean", "std"):
                np.testing.assert_allclose(
                    loc[path].attrs[attr],
                    ref[path].attrs[attr],
                    rtol=1e-5,
                    err_msg=f"{attr} mismatch in {path}",
                )
