from pathlib import Path

import numpy as np
import ray
import torch
from torch_brain.data import Interval
from torch_brain.datasets import DatasetIndex
from tqdm import tqdm

from core.features import compute_isi_histogram
from core.utils.ray_map import imap_unordered
from core.utils.shard_cache import ShardedCache
from ts3.ts3_dataset import IBLBrainWideBenchTS3

CACHE_SAVE_INTERVAL = 25


def _trial_intervals(dataset) -> dict[str, Interval]:
    """Task-aligned windows per session, which is what a LOLCAT snippet is cut from.

    Whole-session sampling is what the rest of TS3 reads, so this lives here rather than on
    the suite dataset: LOLCAT is the only model that wants trials.
    """
    ans = {}
    for rid in dataset.recording_ids:
        recording = dataset.get_recording(rid)
        assert hasattr(recording, "task_aligned_intervals"), f"{rid} has no task_aligned_intervals"
        ans[rid] = recording.task_aligned_intervals.domain

    return ans


def _extract_recording_arrays(rec, intervals, rid, t_offset) -> dict:
    """Pull raw arrays from a recording object.

    Slicing resets the time origin, so trial bounds are rebased by t_offset onto the
    sliced clock.
    """
    return {
        "spike_timestamps": np.asarray(rec.spikes.timestamps),
        "spike_units": np.asarray(rec.spikes.unit_index),
        "unit_ids": np.asarray(rec.units.id).astype(str),
        "trial_starts": np.asarray(intervals.start, dtype=np.float64) - t_offset,
        "trial_ends": np.asarray(intervals.end, dtype=np.float64) - t_offset,
        "rid": rid,
    }


def _process_recording_arrays(
    spike_timestamps,
    spike_units,
    unit_ids,
    trial_starts,
    trial_ends,
    rid,
    n_bins,
    t_min,
    t_max,
    log_bins=True,
) -> dict | None:
    """Compute per-trial ISI histograms for all units. Safe to run in a Ray worker."""
    try:
        hists_list = []
        uid_list = []
        for i, uid in enumerate(unit_ids):
            spikes = spike_timestamps[spike_units == i]
            hists = []
            for t0, t1 in zip(trial_starts, trial_ends, strict=False):
                window = spikes[(spikes >= t0) & (spikes < t1)]
                hists.append(compute_isi_histogram(window, n_bins, t_min, t_max, log_bins))
            hists_list.append(np.stack(hists).astype(np.float32))
            uid_list.append(uid)
        return {
            "isi_hists": hists_list,
            "uids": uid_list,
            "rid": rid,
        }
    except Exception as e:
        print(f"Failed {rid}: {e}")
        return None


@ray.remote
def _process_recording_remote(**kwargs):
    return _process_recording_arrays(**kwargs)


def _cache_params(n_bins, t_min, t_max, log_bins) -> dict:
    """Params that define what a cached histogram is; a mismatch invalidates the cache."""
    return {
        "n_bins": int(n_bins),
        "t_min": float(t_min),
        "t_max": float(t_max),
        "log_bins": bool(log_bins),
    }


def _pack(results: list[dict], params: dict) -> dict:
    return {
        "isi_hists": [torch.from_numpy(h.copy()) for r in results for h in r["isi_hists"]],
        "uids": np.array([uid for r in results for uid in r["uids"]]),
        "rids": np.array(sorted({r["rid"] for r in results})),
        "params": params,
    }


def _merge(chunks: list[dict], params: dict) -> dict:
    merged = {"isi_hists": [x for c in chunks for x in c["isi_hists"]]}
    merged["uids"] = np.concatenate([c["uids"] for c in chunks])
    merged["rids"] = np.array(sorted({r for c in chunks for r in c["rids"].tolist()}))
    merged["params"] = params
    return merged


def _cache(cache_path: Path, params: dict) -> ShardedCache:
    return ShardedCache(
        cache_path,
        suffix=".pt",
        dump=torch.save,
        load=lambda path: torch.load(path, weights_only=False),
        merge=lambda chunks: _merge(chunks, params),
        keep=lambda chunk: chunk.get("params") == params,
    )


def update_lolcat_cache(data_root, n_bins, t_min, t_max, cache_path, log_bins=True) -> None:
    """Build or incrementally update the LOLCAT ISI histogram cache."""
    params = _cache_params(n_bins, t_min, t_max, log_bins)
    cache = _cache(Path(cache_path), params)

    with cache.lock():
        cache.merge_shards()
    cached_rids: set[str] = set()
    if cache.exists():
        existing = cache.read()
        if existing.get("params") == params:
            cached_rids = set(existing["rids"].tolist())
        else:
            print(f"[cache] params changed {existing.get('params')} -> {params}, rebuilding")
            cache.path.unlink()

    if not ray.is_initialized():
        ray.init(log_to_driver=False, ignore_reinit_error=True)

    for regime in ("pretrain", "eval"):
        dataset = IBLBrainWideBenchTS3(root=str(data_root), regime=regime)
        trial_intervals = _trial_intervals(dataset)
        missing_rids = [rid for rid in dataset.recording_ids if rid not in cached_rids]
        if not missing_rids:
            print(f"[{regime}] cache up to date, skipping")
            continue

        print(f"[{regime}] processing {len(missing_rids)} missing recordings")

        def submit(rid, dataset=dataset, trial_intervals=trial_intervals):
            intervals = trial_intervals[rid]
            if len(intervals.start) == 0:
                return None
            t_start = float(intervals.start[0])
            t_end = float(intervals.end[-1])
            rec = dataset[DatasetIndex(rid, start=t_start, end=t_end)]
            arrays = _extract_recording_arrays(rec, intervals, rid, t_start)
            return _process_recording_remote.remote(
                **arrays, n_bins=n_bins, t_min=t_min, t_max=t_max, log_bins=log_bins
            )

        pending, n_shards = [], 0
        results = imap_unordered(submit, missing_rids)
        for result in tqdm(results, total=len(missing_rids), desc=f"{regime} processing"):
            if result is not None:
                pending.append(result)
            if len(pending) >= CACHE_SAVE_INTERVAL:
                cache.write_shard(n_shards, _pack(pending, params))
                n_shards += 1
                pending = []

        if pending:
            cache.write_shard(n_shards, _pack(pending, params))
        with cache.lock():
            cache.merge_shards()

    print(f"LOLCAT cache at {cache.path}")

    ray.shutdown()


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Build or update LOLCAT ISI histogram cache")
    parser.add_argument("--data_root", required=True)
    parser.add_argument("--cache_path", required=True)
    parser.add_argument("--n_bins", type=int, default=128)
    parser.add_argument("--t_min", type=float, default=1e-3)
    parser.add_argument("--t_max", type=float, default=3.0)
    parser.add_argument("--no_log_bins", action="store_true")
    args = parser.parse_args()

    update_lolcat_cache(
        args.data_root,
        args.n_bins,
        args.t_min,
        args.t_max,
        args.cache_path,
        log_bins=not args.no_log_bins,
    )


if __name__ == "__main__":
    main()
