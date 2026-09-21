# /// brainset-pipeline
# python-version = "3.10"
# dependencies = [
#   "ray<2.42",
#   "numpy==1.24.4",
#   "ONE-api==3.4.0",
#   "ibllib==4.0.0",
#   "scipy==1.15.3",
# ]
# ///

import os

os.environ["ONE_REVISION_LAST_BEFORE"] = "2026-09-03"

import argparse
import hashlib
import importlib.metadata
import logging
import subprocess
import time
from argparse import ArgumentParser
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Literal, NamedTuple, get_args

import h5py
import numpy as np
import pandas as pd
from brainbox.io.one import SessionLoader, SpikeSortingLoader
from iblatlas.atlas import AllenAtlas
from iblatlas.regions import BrainRegions
from ibllib.io.video import get_video_frame, url_from_eid
from one.api import ONE, One
from scipy.ndimage import binary_erosion
from torch_brain.data import (
    ArrayDict,
    BrainsetDescription,
    Data,
    DeviceDescription,
    Interval,
    IrregularTimeSeries,
    RegularTimeSeries,
    SessionDescription,
    SubjectDescription,
    serialize_fn_map,
)
from torch_brain.pipeline import BrainsetPipeline
from utils import decimate_signal, regularize_timeseries, resample_timeseries

logging.basicConfig(level=logging.INFO)


# ------------------------------------------------------------------------- unit filtering contract

UnitFilterType = Literal["probe_qc", "firing_rate", "unit_qc"]
ALL_UNIT_FILTERS = get_args(UnitFilterType)
UNIT_FILTER_CHOICES = ("all", *ALL_UNIT_FILTERS)

# unit filtering parameters
PROBE_QC_LABEL = "PASS"
MIN_FIRING_RATE = 1.0
UNIT_QC_LABEL = 1.0


# Read back by src/core/data/unit_filtering.py, which enforces the per-suite contract. Keep the two
# in sync: this script runs standalone, with its own pinned deps, and cannot import from src.
def resolve_unit_filters(requested: list[str], is_pretrain: bool) -> tuple[str, ...]:
    """Expand ``all`` and drop the filters that do not apply to this regime."""
    names = set(ALL_UNIT_FILTERS) if "all" in requested else set(requested)
    if is_pretrain:
        names.discard("probe_qc")  # probe qc is applied to eval sessions only
    return tuple(sorted(names))


def unit_filtering_label(filters: tuple[str, ...], is_pretrain: bool) -> str:
    """``all_units`` for no filter, ``selected_units`` for the full set, else ``custom``."""
    if not filters:
        return "all_units"
    canonical = set(resolve_unit_filters(["all"], is_pretrain))
    return "selected_units" if set(filters) == canonical else "custom"


def unit_filter_thresholds(filters: tuple[str, ...]) -> str:
    """The thresholds behind the applied filters, so a custom build is self-describing."""
    thresholds = {
        "probe_qc": f"qc_neural == {PROBE_QC_LABEL}",
        "firing_rate": f"firing_rate > {MIN_FIRING_RATE}",
        "unit_qc": f"label == {UNIT_QC_LABEL}",
    }
    return ", ".join(thresholds[f] for f in filters)


# ------------------------------------------------------------------------------------ command line

DEFAULT_API_RETRIES = 3
parser = ArgumentParser()
parser.add_argument("--reprocess", action="store_true")
parser.add_argument("--list-sessions", type=str, default=False)
parser.add_argument(
    "--download-first",
    action=argparse.BooleanOptionalAction,
    default=True,
    help="Whether to download data first. Use --no-download-first to skip.",
)
parser.add_argument(
    "--unit-filter",
    type=str,
    action="append",
    choices=UNIT_FILTER_CHOICES,
    default=[],
    help=(
        "Which unit filter(s) to apply. Pass zero or more: probe_qc, firing_rate, unit_qc, "
        "or 'all' for the full set required by TS2 and TS3. Default is none."
    ),
)
parser.add_argument(
    "--small",
    action="store_true",
    default=False,
    help=(
        "Read EIDs from a small sample list (data/pretrain_eids_small.txt, "
        "data/eval_eids_small.txt) instead of the full ones, for a quick end-to-end test "
        "of the pipeline. An arbitrary sample, not an official subset of the benchmark."
    ),
)
parser.add_argument(
    "--api-retries",
    type=int,
    default=DEFAULT_API_RETRIES,
    help="Number of attempts for each API call to ONE.",
)
parser.add_argument(
    "--example-frame",
    action="store_true",
    default=False,
    help="Whether to extract and store an example video frame. Disabled by default.",
)


# ------------------------------------------------------------------------------- shared vocabulary

DATA_DIR = Path(__file__).resolve().parent / "data"
PROBES_WITHOUT_DATA = ["f936a701-5f8a-4aa1-b7a9-9f8b5b69bc7c"]
TARGET_FS_HZ = 50  # shared grid for all behavior streams
GAP_TOLERANCE_SEC = 0.01  # a longer gap ends a contiguous block
FS_TOLERANCE_HZ = 1.0  # slack when matching an observed fs

# stimulus side, choice and block prior share one encoding, so they compare directly.
SIDE_LEFT = 1
SIDE_RIGHT = 0
NO_CHOICE = 2
BLOCK_UNBIASED = -1  # the opening 0.5 block, dropped before storage
STIMULUS_SIDE_MAP = {"Left": SIDE_LEFT, "Right": SIDE_RIGHT}
CHOICE_MAP = {-1: SIDE_RIGHT, 0: NO_CHOICE, 1: SIDE_LEFT}
BLOCK_PRIOR_MAP = {0.2: SIDE_RIGHT, 0.5: BLOCK_UNBIASED, 0.8: SIDE_LEFT}  # probabilityLeft


# ------------------------------------------------------------------------------------ the pipeline

PRETRAIN_TRAIN_RATIO = 0.9
PRETRAIN_VAL_RATIO = 0.1
EVAL_TRAIN_RATIO = 0.4
EVAL_VAL_RATIO = 0.2
EVAL_TEST_RATIO = 0.4
MIN_SPLIT_GAP_SEC = 20.0  # required gap between two adjacent splits


class Pipeline(BrainsetPipeline):
    brainset_id = "ibl_brain_wide_bench_2026"
    parser = parser

    @classmethod
    def get_manifest(cls, raw_dir: Path, args):
        # use pre-determined list of eids for pretrain and eval
        if args.small:
            pretrain_eids_file = "pretrain_eids_small.txt"
            eval_eids_file = "eval_eids_small.txt"
        else:
            pretrain_eids_file = "pretrain_eids.txt"
            eval_eids_file = "eval_eids.txt"

        qc_df = _get_qc_df()

        def get_manifest_from_file(fh, is_pretrain: bool):
            ret = []
            for line in fh:
                eid = line.strip()
                _qc_df = qc_df[qc_df.eid == eid]
                assert _qc_df.qc_behavior.nunique() == 1
                assert _qc_df.qc_behavior_whisker.nunique() == 1
                assert _qc_df.qc_behavior_paws.nunique() == 1
                assert _qc_df.qc_behavior_licks.nunique() == 1
                assert _qc_df.qc_behavior_wheel.nunique() == 1

                item = {
                    "id": eid,
                    "eid": eid,
                    "is_pretrain": is_pretrain,
                    "qc_behavior": _qc_df.qc_behavior.iloc[0],
                    "qc_behavior_whisker": _qc_df.qc_behavior_whisker.iloc[0],
                    "qc_behavior_paws": _qc_df.qc_behavior_paws.iloc[0],
                    "qc_behavior_licks": _qc_df.qc_behavior_licks.iloc[0],
                    "qc_behavior_wheel": _qc_df.qc_behavior_wheel.iloc[0],
                }
                ret.append(item)
            return ret

        manifest_list = []
        with open(DATA_DIR / pretrain_eids_file) as fh:
            manifest_list.extend(get_manifest_from_file(fh, True))
        with open(DATA_DIR / eval_eids_file) as fh:
            manifest_list.extend(get_manifest_from_file(fh, False))

        manifest = pd.DataFrame(manifest_list).set_index("id")

        # connect to the ONE sdk
        # do this once only, and copy the handle for all manifest items
        # otherwise, an error will be raised from token file being overwritten
        ONE.setup(silent=True)
        one = ONE(
            base_url="https://openalyx.internationalbrainlab.org",
            username="intbrainlab",
            password="international",
            cache_dir=raw_dir,
        )
        manifest["one"] = one

        return manifest

    def download(self, manifest_item):
        if not self.args.download_first:
            self.update_status("Skipped Downloading (--no-download-first flag used)")
            # We still MUST return manifest_item so the process() method
            # has the eid and ONE instance to work with from the local cache!
            # Anything missing from the cache is then fetched lazily by ONE
            # during process(), where the same retry budget applies.
            return manifest_item

        self.update_status("Starting Download")

        _retry_one(
            self._download_from_one,
            manifest_item,
            what="download",
            max_retries=self.args.api_retries,
            on_retry=lambda attempt: self.update_status(
                f"Failed to download from ONE on attempt {attempt}. Waiting before retrying..."
            ),
        )

        self.update_status("Ending Download")

        return manifest_item

    def _download_from_one(self, manifest_item):
        eid = manifest_item.eid
        one: One = manifest_item.one

        self.update_status("Downloading Spikes")
        # download() already retries this whole method, so don't nest a second layer
        for pid, probe_name in _resolve_probes(one, eid, max_retries=1):
            if str(pid) in PROBES_WITHOUT_DATA:
                logging.warning(f"Probe {pid} has no data, skipping")
                continue
            # mock=True skips the ~37 MB Allen volumetric atlas download (template +
            # annotation .nrrd) which the pipeline never uses; only atlas.regions is
            # consumed, to map already-resolved channel IDs to acronyms.
            spike_loader = SpikeSortingLoader(
                pid=pid, one=one, eid=eid, pname=probe_name, atlas=AllenAtlas(mock=True)
            )
            spike_loader.download_spike_sorting()
            one.load_dataset(
                eid,
                "waveforms.templates.npy",
                collection=spike_loader.collection,
                download_only=True,
            )
            spike_loader.download_spike_sorting_object(
                obj="electrodeSites", collection=f"alf/{probe_name}", missing="ignore"
            )

        one_details = one.get_details(eid)
        local_path = one_details["local_path"]
        session_loader = SessionLoader(one=one, session_path=local_path, eid=eid)

        self.update_status("Downloading Trial")
        session_loader.load_trials()

        if manifest_item.qc_behavior_whisker not in ["FAIL", "MISSING"]:
            self.update_status("Downloading Whisker")
            session_loader.load_motion_energy(views=["left"])

        if manifest_item.qc_behavior_paws not in ["FAIL", "MISSING"]:
            self.update_status("Downloading Pose")
            session_loader.load_pose(likelihood_thr=0.0, views=["left"], tracker="lightningPose")

        if manifest_item.qc_behavior_licks not in ["FAIL", "MISSING"]:
            self.update_status("Downloading Lick")
            session_loader.load_licks()

        if manifest_item.qc_behavior_wheel not in ["FAIL", "MISSING"]:
            self.update_status("Downloading Wheel")
            session_loader.load_wheel()

    def process(self, manifest_item):
        eid = manifest_item.eid
        is_pretrain = manifest_item.is_pretrain
        one: One = manifest_item.one

        store_dir = self.processed_dir / "pretrain" if is_pretrain else self.processed_dir / "eval"

        store_dir.mkdir(exist_ok=True, parents=True)

        store_path = store_dir / f"{eid}.h5"

        if store_path.exists() and not self.args.reprocess:
            self.update_status("Skipped Processing")
            return

        self.update_status("Processing session")

        unit_filters = resolve_unit_filters(self.args.unit_filter, is_pretrain)
        filtering_label = unit_filtering_label(unit_filters, is_pretrain)
        logging.info(
            f"Unit filtering: {filtering_label} "
            f"(filters applied: {', '.join(unit_filters) or 'none'})"
        )

        brainset_description = BrainsetDescription(
            id="ibl_brain_wide_bench_2026",
            origin_version="0.0.1",
            # Public from 0.1.0 on, and bumped with CHANGELOG.md from there: the bucket
            # is regenerated under the same keys, so the label is all a downloader has.
            derived_version="0.1.0",
            source="one-api",
            unit_filtering=filtering_label,
            unit_filters=",".join(unit_filters),
            unit_filter_thresholds=unit_filter_thresholds(unit_filters),
            description=(
                "A key challenge in neuroscience is understanding how neurons in hundreds of interconnected brain regions integrate sensory inputs with "
                "previous expectations to initiate movements and make decisions. It is difficult to meet this challenge if different laboratories apply "
                "different analyses to different recordings in different regions during different behaviours. Here we report a comprehensive set of recordings "
                "from 621,733 neurons recorded with 699 Neuropixels probes across 139 mice in 12 laboratories. The data were obtained from mice performing "
                "a decision-making task with sensory, motor and cognitive components. The probes covered 279 brain areas in the left forebrain and midbrain "
                "and the right hindbrain and cerebellum. We provide an initial appraisal of this brain-wide map and assess how neural activity encodes key "
                "task variables. Representations of visual stimuli transiently appeared in classical visual areas after stimulus onset and then spread to "
                "ramp-like activity in a collection of midbrain and hindbrain regions that also encoded choices. Neural responses correlated with impending "
                "motor action almost everywhere in the brain. Responses to reward delivery and consumption were also widespread. This publicly available "
                "dataset represents a resource for understanding how computations distributed across and within brain areas drive behaviour."
            ),
        )

        # get subject metadata
        one_details = _retry_one(
            one.get_details,
            eid,
            what="get session details",
            max_retries=self.args.api_retries,
            on_retry=lambda attempt: self.update_status(
                f"Failed to get session details on attempt {attempt}. Waiting before retrying..."
            ),
        )
        subject_id = one_details["subject"]

        subject = SubjectDescription(
            id=subject_id,
            species="MUS_MUSCULUS",
        )

        # extract experiment metadata
        recording_date = datetime.fromisoformat(one_details["start_time"]).strftime("%Y%m%d")
        lab_id = one_details["lab"]

        device_id = f"{lab_id}_{subject.id}_{recording_date}"

        # register session
        session_description = SessionDescription(
            id=eid,
            recording_date=datetime.strptime(recording_date, "%Y%m%d"),
        )

        # register device
        device_description = DeviceDescription(
            id=device_id,
            recording_tech="NEUROPIXELS_ARRAY",
        )

        # extract spiking activity
        self.update_status("Extracting Spikes")
        spikes, units, probes = extract_spikes(
            one, eid, unit_filters=unit_filters, max_retries=self.args.api_retries
        )

        # extract behavioral data
        self.update_status("Extracting Behavioral Data")

        session_loader = _retry_one(
            SessionLoader,
            one=one,
            session_path=one_details["local_path"],
            eid=eid,
            what="open session loader",
            max_retries=self.args.api_retries,
        )

        behavior_dict = {}

        if manifest_item.qc_behavior == "PASS":
            assert (
                manifest_item.qc_behavior_whisker == "PASS"
                and manifest_item.qc_behavior_paws == "PASS"
                and manifest_item.qc_behavior_licks == "PASS"
                and manifest_item.qc_behavior_wheel == "PASS"
            ), "If behavior QC is PASS, then all other specific behavior QC must be PASS"

        # Shared 50Hz grid: whisker when present, paws as fallback when whisker fails QC.
        grid_timestamps = None
        # whisker motion energy
        if manifest_item.qc_behavior_whisker not in ["FAIL", "MISSING"]:
            # whisker data is always available and all other behavior timestamps will be aligned to it
            self.update_status("Extracting Whisker Data")
            whisker = extract_whisker(session_loader, max_retries=self.args.api_retries)
            grid_timestamps = whisker.timestamps

            validate_whisker(whisker, expected_fs=TARGET_FS_HZ)
            behavior_dict["whisker"] = whisker

        else:
            logging.warning("Whisker data is skipped because of QC")

        # paws
        if manifest_item.qc_behavior_paws not in ["FAIL", "MISSING"]:
            self.update_status("Extracting Pose Data")
            paws = extract_paws(session_loader, max_retries=self.args.api_retries)

            if grid_timestamps is None:
                grid_timestamps = paws.timestamps

            if paws is not None:
                validate_paws(
                    paws,
                    expected_fs=TARGET_FS_HZ,
                    grid_timestamps=grid_timestamps,
                )
                behavior_dict["paws"] = paws
            else:
                logging.warning("Pose data is not available")

        else:
            logging.warning("Pose data is skipped because of QC")

        # lick
        if (
            manifest_item.qc_behavior_licks not in ["FAIL", "MISSING"]
            and grid_timestamps is not None
        ):
            # note: this file isn't always present (typically because the data from the right camera is not usable)
            self.update_status("Extracting Lick Data")
            licks = extract_licks(
                session_loader, grid_timestamps, max_retries=self.args.api_retries
            )

            if licks is not None:
                validate_licks(
                    licks,
                    expected_fs=TARGET_FS_HZ,
                    grid_timestamps=grid_timestamps,
                )
                behavior_dict["licks"] = licks
            else:
                logging.warning("Lick data is not available")

        else:
            logging.warning("Lick data skipped because of QC")

        # wheel
        if (
            manifest_item.qc_behavior_wheel not in ["FAIL", "MISSING"]
            and grid_timestamps is not None
        ):
            self.update_status("Extracting Wheel Data")
            wheel = extract_wheel(
                session_loader, grid_timestamps=grid_timestamps, max_retries=self.args.api_retries
            )
            if wheel is not None:
                validate_wheel(
                    wheel,
                    expected_fs=TARGET_FS_HZ,
                    grid_timestamps=grid_timestamps,
                )
                behavior_dict["wheel"] = wheel
            else:
                logging.warning("Wheel data is not available")

        else:
            logging.warning("Wheel data skipped because of QC")

        if self.args.example_frame and (
            manifest_item.qc_behavior_whisker not in ["FAIL", "MISSING"]
            or manifest_item.qc_behavior_paws not in ["FAIL", "MISSING"]
            or manifest_item.qc_behavior_licks not in ["FAIL", "MISSING"]
        ):
            self.update_status("Extracting Example Video Frame")
            behavior_dict["video"] = extract_video(eid, one, max_retries=self.args.api_retries)

        # register behavior QC
        self.update_status("Registering Behavior QC")
        behavior_qc = get_behavior_qc(manifest_item)

        # extract trial data
        self.update_status("Extracting Trial and Task Intervals Data")
        trials = load_trials(session_loader, max_retries=self.args.api_retries)

        # extract task-aligned intervals
        task_aligned_intervals = extract_task_aligned_intervals(trials)
        if manifest_item.qc_behavior_paws not in ["FAIL", "MISSING"] and paws is not None:
            task_aligned_intervals = _fill_interval_confidence_counts(task_aligned_intervals, paws)

        # register session
        # Note: it is possible to have a bigger data.domain than spikes.domain
        # as spikes.domain is not necessarly a superset of all the behaviors' domains.
        # We decide to restrain splits and data.domain to the spikes domain for the benchmark
        data = Data(
            brainset=brainset_description,
            subject=subject,
            session=session_description,
            device=device_description,
            lab=lab_id,
            # neural activity
            spikes=spikes,
            units=units,
            probes=probes,
            # stimuli and behavior
            trials=trials,
            task_aligned_intervals=task_aligned_intervals,
            **behavior_dict,
            behavior_qc=behavior_qc,
            domain=spikes.domain,
            provenance=get_provenance(self.args),
        )

        # define splits trial data
        self.update_status("Extracting Splits")

        if is_pretrain:
            train_ratio = PRETRAIN_TRAIN_RATIO
            val_ratio = PRETRAIN_VAL_RATIO
            test_ratio = 0
        else:
            train_ratio = EVAL_TRAIN_RATIO
            val_ratio = EVAL_VAL_RATIO
            test_ratio = EVAL_TEST_RATIO

        train_domain, val_domain, test_domain = get_causal_splits(
            data.domain,
            task_aligned_intervals.domain,
            train_ratio,
            val_ratio,
            test_ratio,
            MIN_SPLIT_GAP_SEC,
        )

        # Keyed by consuming scope: f"{scope}_{split}_domain", f"{scope}_normalize.{key}".
        if is_pretrain:
            data.pretrain_train_domain = train_domain
            data.pretrain_val_domain = val_domain
            data.pretrain_normalize = _get_behavior_normalization(data, train_domain)
        else:
            data.ts1_train_domain = train_domain
            data.ts1_val_domain = val_domain
            data.ts1_test_domain = test_domain
            data.ts1_normalize = _get_behavior_normalization(data, train_domain)

            ts2_train_domain, ts2_val_domain, ts2_test_domain = get_chunked_splits(
                data.domain, task_aligned_intervals.domain, MIN_SPLIT_GAP_SEC
            )
            data.ts2_train_domain = ts2_train_domain
            data.ts2_val_domain = ts2_val_domain
            data.ts2_test_domain = ts2_test_domain

        # define held-out units for ts2 co-smoothing
        assign_ts2_co_smoothing_held_out_units(
            data, is_pretrain=is_pretrain, unit_filters=unit_filters
        )

        # save data to disk
        self.update_status("Storing")
        with h5py.File(store_path, "w") as file:
            data.to_hdf5(file, serialize_fn_map=serialize_fn_map)

        try:
            one.save_cache()  # explicitly save before shutdown
        except Exception as e:
            logging.warning(f"Error saving cache: {e}")


# -------------------------------------------------------------------------------------- session qc

QC_FILENAME = "bwm_qc.csv"


def _get_qc_df(eid: str | None = None) -> pd.DataFrame:
    """Get QC metrics as a dataframe.

    Args:
        eid: If given, return only the metadata for that eid.
            If None, return the entire dataframe
    """
    qc_df = pd.read_csv(DATA_DIR / QC_FILENAME)

    if eid is None:
        return qc_df

    return qc_df[qc_df.eid == eid]


QC_BEHAVIOR_FIELDS = [
    "qc_behavior_whisker",
    "qc_behavior_paws",
    "qc_behavior_licks",
    "qc_behavior_wheel",
    "qc_behavior",  # per session, the most severe of the qc_behavior_{signal} metrics above
]


def get_behavior_qc(manifest_item: NamedTuple):
    """Construct an object to store behavior QC from manifest item."""
    return Data(**{qc_field: getattr(manifest_item, qc_field) for qc_field in QC_BEHAVIOR_FIELDS})


# -------------------------------------------------------------------------------------- one access

ONE_RETRY_WAIT_SEC = 5


def _retry_one(
    fn,
    *args,
    what: str,
    max_retries: int = DEFAULT_API_RETRIES,
    on_retry=None,
    **kwargs,
):
    """Call an ONE-backed loader, retrying transient API and download failures.

    Every fetch goes through here so downloads are retried whether they are
    prefetched by download() or pulled in lazily during process().
    """
    attempts = max(1, max_retries)
    for attempt in range(1, attempts + 1):
        try:
            return fn(*args, **kwargs)
        except Exception as e:
            if attempt == attempts:
                raise RuntimeError(f"Failed to {what} from ONE after {attempts} attempts") from e
            if on_retry is not None:
                on_retry(attempt)
            logging.warning(f"Failed to {what} from ONE on attempt {attempt}: {e}")
            time.sleep(ONE_RETRY_WAIT_SEC)


# ------------------------------------------------------------------------------------------ spikes

# region mappings
BRAIN_REGION_ATLASES = ["allen", "beryl", "cosmos"]
QC_NEURAL_FIELDS = [
    "qc_neural_raw",
    "qc_neural_spikesorting",
    "qc_neural_alignment",
    "qc_neural",
]


def extract_spikes(
    one: One,
    eid: str,
    unit_filters: tuple[UnitFilterType, ...],
    max_retries: int = DEFAULT_API_RETRIES,
):
    """Load and reindex spike-sorting outputs across probes for one session."""
    probes_in_order = _resolve_probes(one, eid, max_retries=max_retries)

    spikes_list, units_list, templates_list = [], [], []
    unit_offset = 0

    probe_qc_df = _get_qc_df(eid).set_index("probe_id")

    # Load per-probe spikes and clusters, then concatenate them into
    # a session-level spike stream with globally unique unit indices.
    for pid, probe_name in probes_in_order:
        # mock=True skips the unused ~37 MB Allen volumetric atlas download; only
        # atlas.regions is needed to map resolved channel IDs to acronyms.
        spike_loader = _retry_one(
            SpikeSortingLoader,
            pid=pid,
            one=one,
            eid=eid,
            pname=probe_name,
            atlas=AllenAtlas(mock=True),
            what="open spike sorting loader",
            max_retries=max_retries,
        )
        spikes, clusters, channels = _retry_one(
            spike_loader.load_spike_sorting, what="load spike sorting", max_retries=max_retries
        )

        if len(clusters) == 0:
            # A small number of known probe insertions are expected to have no cluster output;
            # this guards against silently dropping unexpected cases.
            assert str(pid) in PROBES_WITHOUT_DATA, f"Pid {pid} has no clusters"
            continue

        # Merge cluster metadata for unit-level annotations.
        clusters = SpikeSortingLoader.merge_clusters(
            spikes, clusters, channels, compute_metrics=False
        )

        # raw_waveforms() would also fetch waveforms.traces (~3 GB per probe, 1.6 TB over
        # the manifest) to memmap a file nothing reads; .templates is a memmap of this one.
        templates_path = _retry_one(
            one.load_dataset,
            eid,
            "waveforms.templates.npy",
            collection=spike_loader.collection,
            download_only=True,
            what="load waveform templates",
            max_retries=max_retries,
        )
        templates = np.load(templates_path)[clusters.cluster_id]
        templates_list.append(templates)

        # Offset probe-local cluster IDs so unit indices are unique across probes.
        spikes["clusters"] += unit_offset
        spikes_list.append(spikes)

        clusters["probe_id"] = str(pid)
        for field in QC_NEURAL_FIELDS:
            clusters[field] = str(probe_qc_df.loc[str(pid)][field])

        for atlas in BRAIN_REGION_ATLASES:
            clusters[f"region_{atlas}"] = BrainRegions().acronym2acronym(
                clusters["acronym"], mapping=atlas.capitalize()
            )

        # `channels` and `rawInd` stay probe-local (they are the peak channel's site
        # index on its own probe), so they are only unique paired with `probe_id`.
        units_list.append(pd.DataFrame(clusters).rename(columns={"uuids": "id"}))

        # Advance index offset for the next probe.
        num_units = len(clusters["cluster_id"])
        unit_offset += num_units

        assert num_units == len(np.unique(clusters["cluster_id"])), "There are duplicate units"

        assert num_units == len(np.unique(spikes["clusters"])), (
            "There are units that have no spikes"
        )

    timestamps = np.concatenate([s["times"] for s in spikes_list])
    unit_index = np.concatenate([s["clusters"] for s in spikes_list])
    amplitude = np.concatenate([s["amps"] for s in spikes_list])
    depth = np.concatenate([s["depths"] for s in spikes_list])

    # Store units.
    units = ArrayDict.from_dataframe(
        pd.concat(units_list, ignore_index=True),
        unsigned_to_long=True,
    )
    units.wf_template = np.concatenate(templates_list, axis=0)
    units.location = np.stack([units.x, units.y, units.z], axis=1)

    # Store probes.
    probes_dict = defaultdict(list)
    for pid, probe_name in probes_in_order:
        pid = str(pid)
        if pid in PROBES_WITHOUT_DATA:
            continue
        probes_dict["id"].append(pid)
        probes_dict["name"].append(probe_name)
        for field in QC_NEURAL_FIELDS:
            probes_dict[field].append(str(probe_qc_df.loc[pid][field]))

    probes_df = pd.DataFrame(probes_dict)
    probes = ArrayDict.from_dataframe(probes_df, unsigned_to_long=True)

    # Apply unit filters
    num_total_units = len(units)
    unit_mask = np.ones(num_total_units, dtype=bool)
    if "probe_qc" in unit_filters:  # resolve_unit_filters already dropped it for pretrain
        unit_mask &= units.qc_neural == PROBE_QC_LABEL
    if "firing_rate" in unit_filters:
        unit_mask &= units.firing_rate > MIN_FIRING_RATE
    if "unit_qc" in unit_filters:
        unit_mask &= units.label == UNIT_QC_LABEL
    num_valid_units = unit_mask.sum()
    logging.info(f"Keeping {num_valid_units} units out of {num_total_units} after unit filters")

    # filter units
    units = units.select_by_mask(unit_mask)

    # remap unit indices
    valid_unit_indices = np.where(unit_mask)[0]
    unit_index_remap = np.zeros(num_total_units, dtype=int)
    unit_index_remap[unit_mask] = np.arange(num_valid_units)

    # filter spikes
    spike_mask = np.isin(unit_index, valid_unit_indices)
    timestamps = timestamps[spike_mask]
    unit_index = unit_index_remap[unit_index[spike_mask]]
    amplitude = amplitude[spike_mask]
    depth = depth[spike_mask]

    # Store spikes and sort the timestamps.
    spikes = IrregularTimeSeries(
        timestamps=timestamps,
        unit_index=unit_index,
        amplitude=amplitude,
        depth=depth,
        domain="auto",
    )
    spikes.sort()

    # Unit UUIDs should be globally unique after concatenation.
    assert len(units.id) == len(np.unique(units.id)), "uuids is not unique"

    return spikes, units, probes


def _resolve_probes(
    one: One, eid: str, max_retries: int = DEFAULT_API_RETRIES
) -> list[tuple[str, str]]:
    """A session's (pid, probe_name) pairs, ordered by probe name."""
    pids, probe_names = _retry_one(
        one.eid2pid, eid, what="resolve probe insertions", max_retries=max_retries
    )
    return sorted(zip(pids, probe_names, strict=True), key=lambda p: (p[1], str(p[0])))


# -------------------------------------------------------------------------------- behavior signals

WHISKER_ALLOWED_FS_HZ = [60, 150]  # source video rates; anything else is a bug


def extract_whisker(
    session_loader: SessionLoader,
    return_full: bool = False,
    max_retries: int = DEFAULT_API_RETRIES,
):
    """Load whisker motion energy, regularize timestamps, and resample to the target fs."""
    _retry_one(
        session_loader.load_motion_energy,
        views=["left"],
        what="load motion energy",
        max_retries=max_retries,
    )

    df_motion_energy = session_loader.motion_energy["leftCamera"]
    raw_timestamps = df_motion_energy.times.values
    motion_energy = df_motion_energy.whiskerMotionEnergy.values

    assert len(raw_timestamps) == len(motion_energy), (
        f"Number of timestamps ({len(raw_timestamps)}) doesn't match number of motion energy values ({len(motion_energy)})"
    )

    whisker = IrregularTimeSeries(
        timestamps=raw_timestamps, whisker_motion_energy=motion_energy, domain="auto"
    )

    # First, regularize the timestamp grid.
    regularized_whisker = regularize_timeseries(whisker, gap_tol=GAP_TOLERANCE_SEC)
    whisker_fs = 1 / np.median(np.diff(regularized_whisker.timestamps))

    # The source video is expected at 60 Hz or 150 Hz.
    assert any(np.abs(whisker_fs - fs) < FS_TOLERANCE_HZ for fs in WHISKER_ALLOWED_FS_HZ), (
        f"Whisker fs is not one of {WHISKER_ALLOWED_FS_HZ}Hz (or within {FS_TOLERANCE_HZ}Hz), it is {whisker_fs}"
    )

    # Then resample to the shared target fs (50 Hz).
    resampled_whisker = resample_timeseries(regularized_whisker, target_fs=TARGET_FS_HZ)

    # Return the resampled trace and raw timestamps for cross-modal alignment.
    if not return_full:
        return resampled_whisker

    return {
        "raw": whisker,
        "regularized": regularized_whisker,
        "resampled": resampled_whisker,
        "regularized_fs": whisker_fs,
        "resampled_fs": TARGET_FS_HZ,
    }


def validate_whisker(whisker, expected_fs):
    assert np.allclose(np.diff(whisker.timestamps), 1 / expected_fs), (
        "Whisker timestamps are not at the expected fs"
    )


# Keys are the ANIMAL's paws, the columns are named for the IMAGE side, so the crossover
# is deliberate: the paw on the left of the frame (paw_l_*) is the animal's right paw.
LEFT_PAW_KEYS = {
    "pos": ["paw_r_x", "paw_r_y"],
    "likelihood": "paw_r_likelihood",
}
RIGHT_PAW_KEYS = {
    "pos": ["paw_l_x", "paw_l_y"],
    "likelihood": "paw_l_likelihood",
}
PAWS_ALLOWED_FS_HZ = [60, 150]
PAW_LIKELIHOOD_THRESHOLD = 0.9


def extract_paws(
    session_loader: SessionLoader,
    return_full: bool = False,
    max_retries: int = DEFAULT_API_RETRIES,
):
    """Load paws keypoints, regularize timestamps, and resample to the target fs."""
    _retry_one(
        session_loader.load_pose,
        likelihood_thr=0.0,
        views=["left"],
        tracker="lightningPose",
        what="load pose",
        max_retries=max_retries,
    )
    pose_df = session_loader.pose["leftCamera"]
    if pose_df is None:
        return None

    raw_timestamps = pose_df.times.values

    # Extract paw keypoints (exclude nose, pupil, tube and tongue keypoints).
    # The tube is static, and tongue dynamics are represented by lick events.
    # Build paws time series on the video timestamp axis.
    paws = IrregularTimeSeries(
        timestamps=raw_timestamps,
        # paws
        left_paw=pose_df[LEFT_PAW_KEYS["pos"]].to_numpy(),
        left_paw_likelihood=pose_df[LEFT_PAW_KEYS["likelihood"]].to_numpy(),
        right_paw=pose_df[RIGHT_PAW_KEYS["pos"]].to_numpy(),
        right_paw_likelihood=pose_df[RIGHT_PAW_KEYS["likelihood"]].to_numpy(),
        domain="auto",
    )

    # First, regularize both streams to a nearly uniform timestamp grid.
    regularized_paws = regularize_timeseries(paws, gap_tol=GAP_TOLERANCE_SEC)
    paws_fs = 1.0 / np.median(np.diff(regularized_paws.timestamps))

    # Validate the observed fs against the nominal video frame rates.
    assert any(np.abs(paws_fs - fs) < FS_TOLERANCE_HZ for fs in PAWS_ALLOWED_FS_HZ), (
        f"Paw fs is not one of {PAWS_ALLOWED_FS_HZ}Hz (or within {FS_TOLERANCE_HZ}Hz), it is {paws_fs}"
    )

    # Derive per-axis velocity for key tracked keypoints.
    dt = 1 / paws_fs
    regularized_paws.left_paw_v_xy = np.gradient(regularized_paws.left_paw, dt, axis=0)
    regularized_paws.left_paw_speed = np.linalg.norm(regularized_paws.left_paw_v_xy, axis=1)
    regularized_paws.right_paw_v_xy = np.gradient(regularized_paws.right_paw, dt, axis=0)
    regularized_paws.right_paw_speed = np.linalg.norm(regularized_paws.right_paw_v_xy, axis=1)

    # Resample both streams to the common 50 Hz target fs.
    resampled_paws = resample_timeseries(regularized_paws, target_fs=TARGET_FS_HZ)

    left_mask = resampled_paws.left_paw_likelihood >= PAW_LIKELIHOOD_THRESHOLD
    right_mask = resampled_paws.right_paw_likelihood >= PAW_LIKELIHOOD_THRESHOLD
    # need to erode the mask as the np.gradient use central differences (Second-Order Accurate)
    resampled_paws.is_left_paw_confident = binary_erosion(left_mask, border_value=1)
    resampled_paws.is_right_paw_confident = binary_erosion(right_mask, border_value=1)

    if not return_full:
        return resampled_paws

    return {
        "raw": paws,
        "regularized": regularized_paws,
        "resampled": resampled_paws,
        "regularized_fs": paws_fs,
        "resampled_fs": TARGET_FS_HZ,
    }


def validate_paws(paws, expected_fs, grid_timestamps):
    assert np.allclose(np.diff(paws.timestamps), 1 / expected_fs), (
        "Paw timestamps are not at the expected fs"
    )
    assert np.allclose(paws.timestamps, grid_timestamps), (
        "Paw timestamps do not match the reference grid"
    )


def extract_licks(
    session_loader: SessionLoader,
    grid_timestamps: np.ndarray,
    return_full: bool = False,
    max_retries: int = DEFAULT_API_RETRIES,
):
    """Load lick timestamps, bin them on the reference grid, and return a licking rate."""
    try:
        _retry_one(session_loader.load_licks, what="load licks", max_retries=max_retries)
    except Exception:
        return None
    lick_timestamps = session_loader.licks.times.to_numpy()

    domain_start = grid_timestamps[0]
    binned_licking = np.zeros_like(grid_timestamps, dtype=np.int32)
    bin_index = np.floor((lick_timestamps - domain_start) * TARGET_FS_HZ).astype(int)
    valid = (bin_index >= 0) & (bin_index < len(binned_licking))
    np.add.at(binned_licking, bin_index[valid], 1)

    # Convert per-bin counts to an approximate per-second rate.
    rate = binned_licking * float(TARGET_FS_HZ)

    licks = RegularTimeSeries(
        sampling_rate=TARGET_FS_HZ,
        domain_start=domain_start,
        licking_rate=rate,
    )

    if not return_full:
        return licks

    return {
        "raw": lick_timestamps,
        "regularized": binned_licking,
        "resampled": licks,
        "regularized_fs": None,
        "resampled_fs": TARGET_FS_HZ,
    }


def validate_licks(licks, expected_fs, grid_timestamps):
    assert np.allclose(np.diff(licks.timestamps), 1 / expected_fs), (
        "Lick timestamps are not at the expected fs"
    )
    assert np.allclose(licks.timestamps, grid_timestamps), (
        "Lick timestamps do not match the reference grid"
    )


WHEEL_INIT_FS_HZ = 1_000.0
WHEEL_DOWNSAMPLE_FACTORS = [10, 2]  # 1kHz -> 100Hz -> 50Hz


def extract_wheel(
    session_loader: SessionLoader,
    grid_timestamps: np.ndarray,
    return_full: bool = False,
    max_retries: int = DEFAULT_API_RETRIES,
):
    """Load wheel data, regularize timestamps, and resample to the target fs."""
    _retry_one(session_loader.load_wheel, what="load wheel", max_retries=max_retries)

    timestamps = session_loader.wheel["times"].to_numpy().astype(np.float64)
    pos = session_loader.wheel["position"].to_numpy()
    vel = session_loader.wheel["velocity"].to_numpy()

    wheel = IrregularTimeSeries(timestamps=timestamps, wheel_pos=pos, wheel_vel=vel, domain="auto")

    # Before resampling the wheel data, it need to align timestamps to the target timestamps

    def snap_interval_round(timestamps, target_fs, t0):
        dt = 1.0 / target_fs
        # Push start forward to the next grid point
        start = t0 + np.ceil((timestamps[0] - t0) / dt) * dt
        # Pull end back to the previous grid point
        end = t0 + np.floor((timestamps[-1] - t0) / dt) * dt
        return start, end

    start_timestamp, end_timestamp = snap_interval_round(
        timestamps, TARGET_FS_HZ, grid_timestamps[0]
    )

    regularized_wheel = regularize_timeseries(
        wheel,
        gap_tol=GAP_TOLERANCE_SEC,
        target_fs=WHEEL_INIT_FS_HZ,
        start_timestamp=start_timestamp,
        end_timestamp=end_timestamp,
    )

    # Decimate from 1kHz to 100Hz, then 100Hz to 50Hz
    pos = decimate_signal(regularized_wheel.wheel_pos, WHEEL_DOWNSAMPLE_FACTORS, WHEEL_INIT_FS_HZ)
    vel = decimate_signal(regularized_wheel.wheel_vel, WHEEL_DOWNSAMPLE_FACTORS, WHEEL_INIT_FS_HZ)

    domain_start = regularized_wheel.timestamps[0]
    resampled_wheel = RegularTimeSeries(
        sampling_rate=TARGET_FS_HZ,
        domain_start=domain_start,
        wheel_pos=pos,
        wheel_vel=vel,
        wheel_speed=np.abs(vel),
    )

    if not return_full:
        return resampled_wheel

    return {
        "raw": wheel,
        "regularized": regularized_wheel,
        "resampled": resampled_wheel,
        "regularized_fs": WHEEL_INIT_FS_HZ,
        "resampled_fs": TARGET_FS_HZ,
    }


def validate_wheel(wheel, expected_fs, grid_timestamps, eps=1e-9):
    assert np.allclose(np.diff(wheel.timestamps), 1 / expected_fs), (
        "Wheel timestamps are not at the expected fs"
    )
    # Compare only where the two domains overlap; eps absorbs float error at the edges.
    overlap_start = max(wheel.timestamps[0], grid_timestamps[0]) + eps
    overlap_end = min(wheel.timestamps[-1], grid_timestamps[-1]) + eps

    inter_wheel = wheel.timestamps[
        (overlap_start <= wheel.timestamps) & (wheel.timestamps <= overlap_end)
    ]
    inter_grid = grid_timestamps[
        (overlap_start <= grid_timestamps) & (grid_timestamps <= overlap_end)
    ]
    assert np.allclose(inter_wheel, inter_grid), (
        "Wheel and reference grid domains do not intersect at the same times"
    )


def extract_video(eid: str, one: One, max_retries: int = DEFAULT_API_RETRIES) -> Data:
    """Extract an example frame from the left camera and return it in a Data object."""
    example_frame = None
    try:
        video_url = _retry_one(
            url_from_eid, eid, one=one, what="resolve video url", max_retries=max_retries
        )["left"]
        example_frame = _retry_one(
            get_video_frame,
            video_url,
            frame_number=1_000,
            what="stream left video",
            max_retries=max_retries,
        )[:, :, 0]
    except (KeyError, RuntimeError):
        logging.warning("Could not load video frame for left cam.")

    return Data(example_frame=example_frame)


# ------------------------------------------------------------------------------------------ trials

TRIAL_KEYS_REMAPPING = {
    "intervals_0": "start",  # start of trial
    "stimOn_times": "stim_on_time",  # visual stimulus appearance
    "goCueTrigger_times": "go_cue_trigger_time",  # audio command sent
    "goCue_times": "go_cue_time",  # audio sound plays
    "firstMovement_times": "movement_onset_time",  # reaction start
    "response_times": "response_time",  # wheel threshold crossing
    "feedback_times": "feedback_time",  # outcome delivery
    "stimOff_times": "stim_off_time",  # visual stimulus removal
    "intervals_1": "end",  # end of trial
    "probabilityLeft": "probability_left",
    "feedbackType": "feedback_type",
    "rewardVolume": "reward_volume",
    "contrastLeft": "contrast_left",
    "contrastRight": "contrast_right",
}
TRIAL_TIMEKEYS = [
    "start",
    "end",
    "stim_on_time",
    "feedback_time",
    "go_cue_trigger_time",
    "go_cue_time",
    "stim_off_time",
    "response_time",
    "movement_onset_time",
]
CONTRAST_MAP = {0.0: 0, 0.0625: 1, 0.125: 2, 0.25: 3, 1.0: 4}
REWARD_MAP = {-1: 0, 1: 1}
MIN_RT_SEC = 0.0  # excludes movement starting before stimulus onset
MAX_RT_SEC = 10.0
REQUIRED_TRIAL_COLUMNS = [
    "stimOn_times",
    "choice",
    "feedback_times",
    "probabilityLeft",
    "firstMovement_times",
    "feedbackType",
]

# Reasons to DROP a trial, OR-ed; callers negate. Runs on the RAW ONE dataframe, where
# `choice == 0` is no-choice, not SIDE_RIGHT as it becomes after CHOICE_MAP.
EXCLUDED_TRIALS_QUERY = " | ".join(
    [
        f"(firstMovement_times - stimOn_times < {MIN_RT_SEC})",
        f"(firstMovement_times - stimOn_times > {MAX_RT_SEC})",
        *(f"{column}.isnull()" for column in REQUIRED_TRIAL_COLUMNS),
        "(choice == 0)",
    ]
)


def load_trials(session_loader: SessionLoader, max_retries: int = DEFAULT_API_RETRIES):
    """Load trials data, remap columns, convert to Interval objects, and filter trials."""
    _retry_one(session_loader.load_trials, what="load trials", max_retries=max_retries)
    trials = session_loader.trials

    trials = trials.rename(columns=TRIAL_KEYS_REMAPPING)

    # relevant columns: choice, probability_left, ...
    trials = Interval.from_dataframe(trials, timekeys=TRIAL_TIMEKEYS)

    # choice is -1, 0 or 1, mapped to SIDE_RIGHT, NO_CHOICE and SIDE_LEFT
    trials.choice = pd.Series(trials.choice).map(CHOICE_MAP).to_numpy()

    # feedback -1 is 0 and feedback 1 is 1
    trials.reward = pd.Series(trials.feedback_type).map(REWARD_MAP).to_numpy()

    # probabilityLeft is 0.2, 0.5 or 0.8, mapped to SIDE_RIGHT, BLOCK_UNBIASED and SIDE_LEFT
    trials.block_prior = pd.Series(trials.probability_left).map(BLOCK_PRIOR_MAP).to_numpy()

    # SIDE_RIGHT if right, SIDE_LEFT if left
    side_labels = np.where(np.isnan(trials.contrast_left), "Right", "Left")
    trials.stimulus_side = pd.Series(side_labels).map(STIMULUS_SIDE_MAP).to_numpy()

    # sanity check: the encoded task variables are compatible
    assert np.all(
        # rewarded iff the animal made a choice and it matches the stimulus side
        ((trials.choice != NO_CHOICE) & (trials.choice == trials.stimulus_side)) == trials.reward
    ), "Choice, stimulus side and reward are not mutually consistent"

    # left and right contrasts are 0., 0.0625, 0.125, 0.25, 1.0
    contrast = np.nan_to_num(trials.contrast_left) + np.nan_to_num(trials.contrast_right)
    trials.stimulus_contrast = pd.Series(contrast).map(CONTRAST_MAP).to_numpy()

    trials.is_valid = ~session_loader.trials.eval(EXCLUDED_TRIALS_QUERY).values

    assert trials.is_disjoint(), "Trials are not disjoint"
    trials.sort()

    # join key for the task-aligned intervals
    trials.trial_index = np.arange(len(trials))

    return trials


# {EVENT}_{PRE,POST}_WINDOW_SEC: offset from that trial event, PRE back, POST forward.
STIM_ON_PRE_WINDOW_SEC = 0.9
STIM_ON_POST_WINDOW_SEC = 0.1
BLOCK_PRIOR_PRE_WINDOW_SEC = 1.0  # back from stimulus onset, not from a block boundary
MOVEMENT_PRE_WINDOW_SEC = 1.0  # the choice window
MOVEMENT_POST_WINDOW_SEC = 1.0
FEEDBACK_POST_WINDOW_SEC = 1.0


def extract_task_aligned_intervals(trials: Interval):
    """Extract task-aligned intervals from trials data."""
    trial_index = np.arange(len(trials))
    trial_index = trial_index[trials.is_valid]
    trials = trials.select_by_mask(trials.is_valid)

    # stimulus side
    stimulus_side = Interval(
        start=trials.stim_on_time - STIM_ON_PRE_WINDOW_SEC,
        end=trials.stim_on_time + STIM_ON_POST_WINDOW_SEC,
        stimulus_side=trials.stimulus_side,
        trial_index=trial_index,
    )
    # stimulus contrast
    stimulus_contrast = Interval(
        start=trials.stim_on_time - STIM_ON_PRE_WINDOW_SEC,
        end=trials.stim_on_time + STIM_ON_POST_WINDOW_SEC,
        stimulus_contrast=trials.stimulus_contrast,
        trial_index=trial_index,
    )

    # choice
    choice = Interval(
        start=trials.movement_onset_time - MOVEMENT_PRE_WINDOW_SEC,
        end=trials.movement_onset_time,
        choice=trials.choice,
        trial_index=trial_index,
    )

    # reward
    reward = Interval(
        start=trials.feedback_time,
        end=trials.feedback_time + FEEDBACK_POST_WINDOW_SEC,
        reward=trials.reward,
        trial_index=trial_index,
    )

    # movement intervals
    movement_window = Interval(
        start=trials.movement_onset_time,
        end=trials.movement_onset_time + MOVEMENT_POST_WINDOW_SEC,
        trial_index=trial_index,
    )

    # licks intervals
    licking_window = Interval(
        start=trials.feedback_time,
        end=trials.feedback_time + FEEDBACK_POST_WINDOW_SEC,
        trial_index=trial_index,
    )

    # NOT row-aligned with the other intervals: unbiased trials carry no prior, dropped.
    block_prior = Interval(
        start=trials.stim_on_time - BLOCK_PRIOR_PRE_WINDOW_SEC,
        end=trials.stim_on_time,
        block_prior=trials.block_prior,
        trial_index=trial_index,
    )
    block_prior_mask = block_prior.block_prior != BLOCK_UNBIASED
    block_prior = block_prior.select_by_mask(block_prior_mask)

    # sanity check: every unbiased (p_left = 0.5) trial has been dropped
    assert np.all(block_prior.block_prior != BLOCK_UNBIASED), "Some blocks have a prior of 0.5"

    tasks = [
        stimulus_side,
        stimulus_contrast,
        choice,
        movement_window,
        licking_window,
        reward,
    ]

    # Get the length from the first valid attribute
    num_indices = len(tasks[0])
    assert all(len(t) == num_indices for t in tasks), "Tasks have mismatched lengths"
    assert len(block_prior_mask) == num_indices, "Block prior mask has mismatched length"

    domain_starts = np.full(num_indices, np.inf)
    domain_ends = np.full(num_indices, -np.inf)

    for task in tasks:
        # Using [:] to force loading from HDF5 into memory
        domain_starts = np.minimum(domain_starts, task.start[:])
        domain_ends = np.maximum(domain_ends, task.end[:])

    domain_starts[block_prior_mask] = np.minimum(
        domain_starts[block_prior_mask], block_prior.start[:]
    )
    domain_ends[block_prior_mask] = np.maximum(domain_ends[block_prior_mask], block_prior.end[:])

    if np.any(domain_starts == np.inf) or np.any(domain_ends == -np.inf):
        raise AssertionError("Found trials with NO task-aligned data")

    domain = Interval(start=domain_starts, end=domain_ends)

    assert domain.is_disjoint(), "task_aligned_intervals domain is not disjoint"

    task_aligned_intervals = Data(
        block_prior=block_prior,
        stimulus_side=stimulus_side,
        stimulus_contrast=stimulus_contrast,
        choice=choice,
        reward=reward,
        movement_window=movement_window,
        licking_window=licking_window,
        domain=domain,
    )

    return task_aligned_intervals


INTERVAL_CONFIDENCE_COUNT_MASKS = {
    # task-aligned interval name -> the masks to count confident samples for
    "movement_window": ["is_left_paw_confident", "is_right_paw_confident"],
}


def _fill_interval_confidence_counts(task_aligned_intervals: Interval, paws: IrregularTimeSeries):
    """Count the confident paw samples inside each task-aligned interval.

    For each interval defined in INTERVAL_CONFIDENCE_COUNT_MASKS, count how many confident
    paw samples fall within each interval window and store the result as a new attribute
    (e.g. ``num_left_paw_confident``) on the corresponding sub-interval.
    """
    for task_aligned_name, confidence_names in INTERVAL_CONFIDENCE_COUNT_MASKS.items():
        task_aligned_interval = getattr(task_aligned_intervals, task_aligned_name)
        for confidence_name in confidence_names:
            counts_key = f"num_{confidence_name.removeprefix('is_')}"
            confidence_counts = [
                getattr(paws.slice(start, end), confidence_name).sum()
                for start, end in task_aligned_interval
            ]
            setattr(
                task_aligned_interval,
                counts_key,
                np.array(confidence_counts),
            )
    return task_aligned_intervals


# ------------------------------------------------------------------------------------------ splits


def get_causal_splits(
    domain: Interval,
    task_aligned_domain: Interval,
    train_ratio: float = 0.4,
    val_ratio: float = 0.2,
    test_ratio: float = 0.4,
    min_split_gap: float = 0.0,
):
    assert np.isclose(train_ratio + val_ratio + test_ratio, 1.0), "split are not summing to one"

    assert min_split_gap >= 0.0, "min_split_gap must be non-negative"

    start, end = domain.start[0], domain.end[-1]

    # Restraining task_aligned_domain to be a subset of domain
    task_aligned_domain = task_aligned_domain & domain

    task_starts, task_ends = task_aligned_domain.start, task_aligned_domain.end
    task_lengths = task_ends - task_starts

    total_task_length = float(np.sum(task_lengths))
    cumulative_task_lengths = np.cumsum(task_lengths)

    def _timestamp_from_task_offset(offset: float) -> float:
        """Map cumulative task-time offset to a real timestamp.

        The result is snapped to the nearest task interval edge, never inside one.
        """
        offset = float(np.clip(offset, 0.0, total_task_length))
        idx = int(np.searchsorted(cumulative_task_lengths, offset, side="left"))
        idx = min(idx, len(cumulative_task_lengths) - 1)
        prev = 0.0 if idx == 0 else float(cumulative_task_lengths[idx - 1])
        next_ = float(cumulative_task_lengths[idx])

        # Snap to closest edge in cumulative-time space.
        if (offset - prev) <= (next_ - offset):
            return float(task_starts[idx])
        return float(task_ends[idx])

    def _boundary_with_gap(
        boundary: float,
        min_gap: float,
        task_starts: np.ndarray,
        task_ends: np.ndarray,
        task_edges: np.ndarray,
        label: str,
        eps: float = 1e-12,
    ) -> float:
        """Advance a boundary by min_gap.

        Snaps to the next task-interval edge if it would otherwise land inside one.
        """
        gap_end = boundary + min_gap
        if np.any((task_starts < gap_end) & (gap_end < task_ends)):
            shifted_boundary = float(task_edges[task_edges >= gap_end][0])
        else:
            shifted_boundary = float(gap_end) + eps

        assert not np.any((task_starts < shifted_boundary) & (shifted_boundary < task_ends)), (
            f"{label} {shifted_boundary} is inside a task interval"
        )
        assert shifted_boundary >= boundary, (
            f"{label} {shifted_boundary} is not >= boundary {boundary}"
        )
        assert shifted_boundary - boundary >= min_gap, (
            f"{label} - boundary {shifted_boundary - boundary} is not at least min_gap {min_gap}"
        )
        return shifted_boundary

    task_edges = np.sort(np.concatenate([task_starts, task_ends]))

    # Cut points in cumulative task time.
    train_end = _timestamp_from_task_offset(total_task_length * train_ratio)
    val_end = _timestamp_from_task_offset(total_task_length * (train_ratio + val_ratio))

    # Push each cut forward to open the required gap
    val_start = _boundary_with_gap(
        boundary=train_end,
        min_gap=min_split_gap,
        task_starts=task_starts,
        task_ends=task_ends,
        task_edges=task_edges,
        label="val_start",
    )

    # create domains
    train_domain = Interval(start, train_end)

    if test_ratio > 0:
        test_start = _boundary_with_gap(
            boundary=val_end,
            min_gap=min_split_gap,
            task_starts=task_starts,
            task_ends=task_ends,
            task_edges=task_edges,
            label="test_start",
        )

        val_domain = Interval(val_start, val_end)
        test_domain = Interval(test_start, end)
    else:
        val_domain = Interval(val_start, end)
        test_domain = Interval(np.array([]), np.array([]))

    # leakage checks
    assert len(train_domain & val_domain) == 0, "Leakage between train and val"
    assert len(val_domain & test_domain) == 0, "Leakage between val and test"
    assert len(train_domain & test_domain) == 0, "Leakage between train and test"

    # log task lengths within each split
    _train_intervals = train_domain & task_aligned_domain
    _val_intervals = val_domain & task_aligned_domain
    _test_intervals = test_domain & task_aligned_domain
    logging.info(
        f"Total train task length: {(_train_intervals.end - _train_intervals.start).sum()}"
    )
    logging.info(f"Total val task length: {(_val_intervals.end - _val_intervals.start).sum()}")
    logging.info(f"Total test task length: {(_test_intervals.end - _test_intervals.start).sum()}")

    return train_domain, val_domain, test_domain


SPLIT_CHUNK_DURATION_SEC = 300  # 5 min


def get_chunked_splits(
    domain: Interval,
    task_aligned_domain: Interval,
    min_split_gap: float = 0.0,
    tol: float = 1e-6,  # tolerance for checking if the trial contains the end of the neural recording
):
    """Split into 5min chunks, excluding trials that contain the end of the recording.

    A "chunk" is a wall-clock slice, unrelated to an IBL block. Returns the train, val
    and test domains, each the union of its assigned chunks. The pattern below has no
    same-split neighbours, so ``min_split_gap`` buffers every boundary, not just a cut.
    """
    # First, we cut the domain into 5min chunks
    chunks = Interval.arange(
        domain.start[0], domain.end[-1], SPLIT_CHUNK_DURATION_SEC, include_end=True
    )
    # arange() aliases start/end onto one array: a write to end[i] lands on start[i + 1]
    chunks = Interval(start=chunks.start.copy(), end=chunks.end.copy())

    for i, (start, end) in enumerate(task_aligned_domain):
        # Find the chunks that intersect with the trial
        chunk_mask = (chunks.start <= end) & (chunks.end >= start)
        num_overlapping_chunks = np.sum(chunk_mask)

        # Some recordings have known issues with trials not having neural data at the end
        # of the recording. This checks whether the trial contains the end of the neural recording,
        # in which case we can exclude it from the splits.
        if start - tol <= domain.end[-1] <= end + tol:
            logging.warning(
                f"Trial {i} (is_valid) is not included in the train/val/test split because it contains the end of the neural recording"
            )
            assert num_overlapping_chunks == 0, f"num_overlapping_chunks should be 0 for trial {i}"
            continue

        assert num_overlapping_chunks <= 2, (
            f"Trial {i} intersects with {num_overlapping_chunks} chunks, expected 1 or 2"
        )

        # We only need to adjust the chunks if the trial intersects with two chunks
        if num_overlapping_chunks == 2:
            # Find the indices of the two intersecting chunks
            chunk_indices = np.where(chunk_mask)[0]
            first_chunk_idx, second_chunk_idx = chunk_indices[0], chunk_indices[1]

            # Find the chunk with the larger intersection
            first_chunk_intersection = min(end, chunks.end[first_chunk_idx]) - max(
                start, chunks.start[first_chunk_idx]
            )
            second_chunk_intersection = min(end, chunks.end[second_chunk_idx]) - max(
                start, chunks.start[second_chunk_idx]
            )

            if first_chunk_intersection >= second_chunk_intersection:
                chunks.end[first_chunk_idx] = end
                chunks.start[second_chunk_idx] = end
            else:
                chunks.end[first_chunk_idx] = start
                chunks.start[second_chunk_idx] = start

            logging.warning(
                f"Trial {i} intersects with two chunks, adjusting the chunks, new chunk"
                f" lengths are {chunks.end[first_chunk_idx] - chunks.start[first_chunk_idx]} "
                f"and {chunks.end[second_chunk_idx] - chunks.start[second_chunk_idx]}"
            )

    # Snap to a trial start: the domains are intersected, so a cut mid-trial would clip it
    if min_split_gap > 0:
        trial_starts = task_aligned_domain.start[:]
        for i in range(1, len(chunks)):
            later = trial_starts[trial_starts >= chunks.end[i - 1] + min_split_gap]
            chunks.start[i] = min(later[0], chunks.end[i]) if len(later) else chunks.end[i]

    num_chunks = len(chunks)
    train_mask = np.zeros(num_chunks, dtype=bool)
    val_mask = np.zeros(num_chunks, dtype=bool)
    test_mask = np.zeros(num_chunks, dtype=bool)

    # We will use a [train, test, train, val, test] pattern for splitting the data
    # The ratio of data will be 40%/20%/40% for train/val/test
    train_mask[::5] = True
    train_mask[2::5] = True
    val_mask[3::5] = True
    test_mask[1::5] = True
    test_mask[4::5] = True

    non_empty = chunks.start < chunks.end
    train_chunks = chunks.select_by_mask(train_mask & non_empty)
    val_chunks = chunks.select_by_mask(val_mask & non_empty)
    test_chunks = chunks.select_by_mask(test_mask & non_empty)

    # check for leakage
    assert len(train_chunks & val_chunks) == 0, "Train and val chunks overlap"
    assert len(train_chunks & test_chunks) == 0, "Train and test chunks overlap"
    assert len(val_chunks & test_chunks) == 0, "Val and test chunks overlap"

    kept = chunks.select_by_mask(non_empty)
    assert np.all(kept.start[1:] - kept.end[:-1] >= min_split_gap - tol), (
        f"Two adjacent chunks are closer than min_split_gap {min_split_gap}"
    )

    return train_chunks, val_chunks, test_chunks


# Independent hash domains: val and test scramble the same units differently. The odd
# spelling is load-bearing: rewording a salt reassigns held-out units, voiding scores.
TS2_CO_SMOOTHING_TEST_SALT = "ts2_co_smoothing_test"
TS2_CO_SMOOTHING_VAL_SALT = "ts2_co_smoothing_val"


def assign_ts2_co_smoothing_held_out_units(
    data: Data, is_pretrain: bool, unit_filters: tuple[UnitFilterType, ...]
):
    """Mark the ts2 co-smoothing hold-out units, independently for test and val.

    Eval sessions only, and only with the full unit filter set applied, so the pool the
    hold-out is drawn from is the one the eval suites read back. The rule itself is
    :func:`co_smoothing_held_out_mask`.
    """
    if is_pretrain:
        return

    missing = sorted(set(resolve_unit_filters(["all"], is_pretrain)) - set(unit_filters))
    if missing:
        logging.warning(
            f"Skipping ts2_co_smoothing_held_out_units: missing the {', '.join(missing)} "
            f"unit filter(s) (got {unit_filters or 'none'}). This build cannot be used for "
            "TS2 or TS3 eval, rebuild it with --unit-filter all"
        )
        return

    unit_ids = [u.decode() if isinstance(u, bytes) else str(u) for u in data.units.id]

    data.units.ts2_co_smoothing_is_held_out_test = co_smoothing_held_out_mask(
        unit_ids, TS2_CO_SMOOTHING_TEST_SALT
    )
    data.units.ts2_co_smoothing_is_held_out_val = co_smoothing_held_out_mask(
        unit_ids, TS2_CO_SMOOTHING_VAL_SALT
    )


TS2_CO_SMOOTHING_HELD_OUT_RATIO = 0.1


def co_smoothing_held_out_mask(
    unit_ids: list[str], salt: str, ratio: float = TS2_CO_SMOOTHING_HELD_OUT_RATIO
) -> np.ndarray:
    """Hold out the k units ranking first by ``sha256(salt:unit_id)``.

    A pure function of the unit id, so it survives a reordered unit table; the
    previous rule drew positions and silently picked different neurons. Rank rather
    than threshold keeps k exact and never zero. ``hashlib``, not the builtin
    ``hash()``, which is salted per process.
    """
    k = max(1, round(ratio * len(unit_ids)))
    digests = [hashlib.sha256(f"{salt}:{uid}".encode()).hexdigest() for uid in unit_ids]
    order = sorted(range(len(unit_ids)), key=lambda i: digests[i])
    mask = np.zeros(len(unit_ids), dtype=bool)
    mask[order[:k]] = True
    return mask


# ----------------------------------------------------------------------------------- normalization


def _get_behavior_normalization(data: Data, train_domain: Interval):
    """Compute behavior normalization stats over the train split's movement windows."""
    eps = 1e-8
    normalize = Data()

    movement_window = data.task_aligned_intervals.movement_window
    intervals = movement_window & train_domain

    specs = {
        "whisker": ["whisker_motion_energy"],
        "paws": [
            "right_paw_v_xy",
            "left_paw_v_xy",
            "right_paw_speed",
            "left_paw_speed",
        ],
    }

    for modality_name, signal_names in specs.items():
        modality_norm = Data()

        for signal_name in signal_names:
            signal_values = []
            for s, e in intervals:
                interval_slice = data.slice(s, e, reset_origin=False)
                modality = getattr(interval_slice, modality_name, None)
                if modality is None or not hasattr(modality, signal_name):
                    continue
                values = _get_normalization_values(modality, modality_name, signal_name)
                if values.size > 0:
                    signal_values.append(values)

            if signal_values:
                values = np.concatenate(signal_values, axis=0)
                setattr(
                    modality_norm,
                    signal_name,
                    Data(
                        mean=values.mean(axis=0),
                        std=np.maximum(values.std(axis=0), eps),
                    ),
                )

        setattr(normalize, modality_name, modality_norm)

    return normalize


PAW_SIGNAL_CONFIDENCE_MASK = {
    # paws signal name -> the confidence mask that gates it
    "left_paw_v_xy": "is_left_paw_confident",
    "left_paw_speed": "is_left_paw_confident",
    "right_paw_v_xy": "is_right_paw_confident",
    "right_paw_speed": "is_right_paw_confident",
}


def _get_normalization_values(
    modality: RegularTimeSeries,
    modality_name: str,
    signal_name: str,
) -> np.ndarray:
    values = np.asarray(getattr(modality, signal_name))
    if modality_name != "paws":
        return values

    mask_name = PAW_SIGNAL_CONFIDENCE_MASK.get(signal_name)
    if mask_name is None or not hasattr(modality, mask_name):
        return values

    mask = np.asarray(getattr(modality, mask_name), dtype=bool)
    assert mask.shape[0] == values.shape[0], (
        f"Skipping paws confidence mask for {signal_name} due to shape mismatch ({values.shape} vs {mask.shape})."
    )

    masked_values = values[mask]
    return values if masked_values.size == 0 else masked_values


# -------------------------------------------------------------------------------------- provenance


def get_provenance(cli_args) -> Data:
    """Pipeline commit, arguments and the dependency versions that actually ran.

    ``pipeline_dirty`` matters as much as the commit, and ``pipeline_diff_sha256`` tells two
    dirty builds apart. Both are scoped to this directory: the rest of the repo does not
    change what the pipeline produces.
    """
    here = str(Path(__file__).resolve().parent)

    def _version(pkg: str) -> str:
        try:
            return importlib.metadata.version(pkg)
        except Exception:
            return "unknown"

    def _git(*args: str) -> str:
        try:
            return subprocess.check_output(
                ["git", "-C", here, *args], stderr=subprocess.DEVNULL, text=True
            ).strip()
        except Exception:
            return "unknown"

    commit = _git("rev-parse", "HEAD")
    dirty = (
        str(bool(_git("status", "--porcelain", "--", here))) if commit != "unknown" else "unknown"
    )
    diff = _git("diff", "HEAD", "--", here) if dirty == "True" else ""
    pipeline_args = ", ".join(f"{k}={v}" for k, v in sorted(vars(cli_args).items()))
    return Data(
        pipeline_commit=commit,
        pipeline_dirty=dirty,
        pipeline_diff_sha256=hashlib.sha256(diff.encode()).hexdigest()[:12] if diff else "",
        pipeline_args=pipeline_args[:500],
        built_at=datetime.now().astimezone().isoformat(timespec="seconds"),
        one_revision_last_before=os.environ.get("ONE_REVISION_LAST_BEFORE", ""),
        one_api_version=_version("ONE-api"),
        ibllib_version=_version("ibllib"),
        iblatlas_version=_version("iblatlas"),
        numpy_version=_version("numpy"),
        scipy_version=_version("scipy"),
    )
