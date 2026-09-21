import argparse
from pathlib import Path

import npyx.c4 as npyx_c4
import npyx.corr as npyx_corr
import numpy as np
import ray
from torch_brain.data import Data
from torch_brain.datasets import DatasetIndex
from tqdm import tqdm

from core.dataset import UnitQCPolicy, WholeSessionSpikeDataset
from core.utils.logger import get_cli_logger
from core.utils.ray_map import imap_unordered
from core.utils.shard_cache import ShardedCache
from ibl_bwb_eval.tasks import get_ts3_readout_spec

logger = get_cli_logger()

_SPEC = get_ts3_readout_spec("unit_cosmos")

# What Nemo pretrains on. TS3's extractor reads this cache under its own policy, so a
# divergence between the two rebuilds the whole cache.
NEMO_UNIT_QC = UnitQCPolicy(keep_qc_neural=("PASS", "WARNING"))

# Spike train sampling rate (Hz)
FS = 30_000

ACG_WIN_SIZE = 2_000  # window size in samples
ACG_BIN_SIZE = 1  # bin size in samples
ACG_LOG = True  # use log-spaced ACG

CACHE_SAVE_INTERVAL = 25  # sessions buffered before a shard is written
MAX_IN_FLIGHT = 16  # sessions queued in the ray object store at once


def make_acg3d(
    spike_trains: np.ndarray, window_size: int, bin_size: int, log_acg: bool = True
) -> np.ndarray:
    """Compute 3D ACG (firing-rate vs. lag) for a single unit's spike train.

    Falls back to crosscorr_vs_firing_rate if fast_acg3d raises an IndexError,
    and returns zeros if both fail.

    Returns an array of shape (10, n_bins).
    """
    try:
        _, acg_3d = npyx_c4.fast_acg3d(spike_trains, bin_size, window_size)
    except IndexError:
        try:
            _, acg_3d = npyx_corr.crosscorr_vs_firing_rate(
                spike_trains,
                spike_trains,
                bin_size=bin_size,
                win_size=window_size,
            )
        except (IndexError, ValueError):
            acg_3d = np.zeros((10, int(window_size / bin_size + 1)))

    if log_acg:
        acg_3d, _ = npyx_corr.convert_acg_log(acg_3d, bin_size, window_size)

    return acg_3d


def align_template(x: np.ndarray) -> np.ndarray:
    """Align waveform template so its trough is at index 42.
    Pads with zeros if trough too close to edge, crops to fixed
    90-sample window (42 before trough, 48 after).
    """
    min_idx = x.argmin()
    target_min_idx = 64
    pad_left = max(0, target_min_idx - min_idx)
    pad_right = max(0, min_idx - target_min_idx)
    x = np.pad(x, (pad_left, pad_right), mode="constant", constant_values=0)
    x = x[target_min_idx - 42 : target_min_idx + 48]
    return x


def normalize_waveforms(templates: np.ndarray) -> np.ndarray:
    """Extract peak channel, align, and normalize waveform templates.

    Args:
        templates: (n_units, n_channels, n_samples)

    Returns:
        normalized_templates: (n_units, 90)
    """
    ptps = np.ptp(templates, axis=2)
    maxCH = np.nanargmax(ptps, axis=1)
    peak_templates = templates[np.arange(len(maxCH)), maxCH, :]  # (n_units, n_samples)

    normalized = []
    for template in peak_templates:
        template = align_template(template)
        template = template / max(np.abs(template))
        normalized.append(template)

    return np.array(normalized, dtype=np.float32)


def _extract_session_arrays(data: Data) -> dict:
    """Pull raw numpy arrays out of a Data object."""
    return {
        "wf_template": np.array(data.units.wf_template),
        "unit_ids": np.array(data.units.id),
        "regions": np.array(getattr(data.units, _SPEC.region_key)).astype(str),
        "probe_ids": np.array(data.units.probe_id),
        "spike_timestamps": np.array(data.spikes.timestamps),
        "spike_unit_index": np.array(data.spikes.unit_index),
        "session_id": str(data.session.id),
        "subject_id": str(data.subject.id),
    }


def _process_session_arrays(
    wf_template: np.ndarray,
    unit_ids: np.ndarray,
    regions: np.ndarray,
    probe_ids: np.ndarray,
    spike_timestamps: np.ndarray,
    spike_unit_index: np.ndarray,
    session_id: str,
    subject_id: str,
) -> dict | None:
    """Compute waveforms and ACGs from raw arrays. Safe to run in a Ray worker."""
    try:
        n_units = len(unit_ids)
        norm_templates = normalize_waveforms(wf_template)
        mask = np.isin(regions, _SPEC.label_names)  # a whitelist fails closed

        acgs = []
        for i in range(n_units):
            spike_train = spike_timestamps[spike_unit_index == i]
            spike_train_samples = np.round(spike_train * FS).astype(np.int64)
            acg = make_acg3d(spike_train_samples, ACG_WIN_SIZE, ACG_BIN_SIZE, ACG_LOG)
            acgs.append(acg)
        acgs = np.array(acgs, dtype=np.float32)

        return {
            "waveforms": norm_templates[mask],
            "acgs": acgs[mask],
            "regions": regions[mask],
            "uuids": unit_ids[mask],
            "pids": probe_ids[mask],
            "eids": np.full(mask.sum(), session_id),
            "subject_ids": np.full(mask.sum(), subject_id),
        }
    except Exception as e:
        logger.warning(f"Failed {session_id}: {e}")
        return None


@ray.remote
def _process_session_remote(**kwargs) -> dict | None:
    return _process_session_arrays(**kwargs)


def _submit_session(dataset, sampling_intervals: dict, rid: str):
    """Hand one session to a ray worker, or None if it has no waveform templates."""
    interval = sampling_intervals[rid]
    recording = dataset[DatasetIndex(rid, start=interval.start[0], end=interval.end[-1])]
    if not hasattr(recording.units, "wf_template") or recording.units.wf_template is None:
        return None
    return _process_session_remote.remote(**_extract_session_arrays(recording))


def _cache_params(dataset) -> str:
    """What a cached unit is: the QC policy and the label vocabulary that chose it.

    Field by field rather than the policy's repr, so a rename does not invalidate every
    cache on disk.

    The regime is deliberately not part of it. One cache serves both, keyed by session id,
    and the pretrain and eval sessions are disjoint.
    """
    qc = dataset.unit_qc
    return (
        f"qc_neural={sorted(qc.keep_qc_neural)}"
        f" alignment={sorted(qc.keep_qc_neural_alignment)}"
        f" firing_rate={qc.min_firing_rate} label={qc.min_label}"
        f" labels={sorted(_SPEC.label_names)}"
    )


def _params_of(chunk: dict) -> str | None:
    raw = chunk.get("params")
    return None if raw is None else str(np.asarray(raw).item())


def _concat(chunks: list[dict], params: str) -> dict:
    merged = {k: np.concatenate([c[k] for c in chunks]) for k in chunks[0] if k != "params"}
    merged["params"] = params
    return merged


def _read_npz(path: Path) -> dict:
    data = np.load(path, allow_pickle=True)
    return {k: data[k] for k in data.files}


def _cache(cache_path: Path, params: str) -> ShardedCache:
    return ShardedCache(
        cache_path,
        suffix=".npz",
        dump=lambda payload, path: np.savez(path, **payload),
        load=_read_npz,
        merge=lambda chunks: _concat(chunks, params),
        keep=lambda chunk: _params_of(chunk) == params,
    )


def load_cache(cache_path: Path, dataset) -> dict:
    """Read the whole cache under a shared lock, refusing one built for other units."""
    params = _cache_params(dataset)
    data = _cache(cache_path, params).read()
    if _params_of(data) != params:
        raise RuntimeError(
            f"the cache at {cache_path} holds units chosen by {_params_of(data)}, not {params}"
        )

    return data


def update_cache(cache_path: Path, dataset) -> None:
    """Build or incrementally update the npz cache for the given dataset.

    Holds an exclusive lock for the whole build, so every rank of a DDP run may call this:
    the first to arrive builds and the rest find nothing missing.
    """
    params = _cache_params(dataset)
    cache = _cache(cache_path, params)
    with cache.lock():
        if cache.exists() and _params_of(_read_npz(cache.path)) != params:
            logger.warning(f"cache holds other units than {params}, rebuilding {cache.path}")
            cache.path.unlink()

        cache.merge_shards()  # fold in whatever a previous run left behind
        _build_missing(cache, dataset, params)


def _build_missing(cache: ShardedCache, dataset, params: str) -> None:
    cached_eids = set()
    if cache.exists():
        cached_eids = set(_read_npz(cache.path)["eids"].tolist())

    missing_rids = [
        rid
        for rid in dataset.recording_ids
        if dataset.get_recording(rid).session.id not in cached_eids
    ]
    if not missing_rids:
        return

    started_ray = not ray.is_initialized()
    if started_ray:
        ray.init(log_to_driver=False)

    logger.info(f"Processing {len(missing_rids)} sessions...")
    sampling_intervals = dataset.get_sampling_intervals()

    pending: list[dict] = []
    n_shards = 0
    results = imap_unordered(
        lambda rid: _submit_session(dataset, sampling_intervals, rid),
        missing_rids,
        max_in_flight=MAX_IN_FLIGHT,
    )
    for result in tqdm(results, total=len(missing_rids), desc="Processing sessions"):
        if result is not None:
            pending.append(result)
        if len(pending) >= CACHE_SAVE_INTERVAL:
            cache.write_shard(n_shards, _concat(pending, params))
            n_shards += 1
            pending = []

    if pending:
        cache.write_shard(n_shards, _concat(pending, params))
    cache.merge_shards()
    logger.info(f"Cache saved to {cache.path}")

    if started_ray:
        ray.shutdown()


def main():
    parser = argparse.ArgumentParser(description="Build the NEMO pretrain cache")
    parser.add_argument("--data_root", required=True)
    parser.add_argument("--cache_path", required=True)
    args = parser.parse_args()

    dataset = WholeSessionSpikeDataset(
        unit_qc=NEMO_UNIT_QC,
        root=args.data_root,
        regime="pretrain",
    )
    update_cache(Path(args.cache_path), dataset)


if __name__ == "__main__":
    main()
