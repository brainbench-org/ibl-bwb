"""The training-free ISI baseline.

Inductive, and the cheapest possible demonstration of why: the embedding is a statistic of
the unit's own spike train, so a held-out unit needs no weights to represent it and there is
nothing to adapt on the eval sessions. The histogram itself is model-agnostic and lives in
``core.features``, where LOLCAT reads it too.
"""

import numpy as np
import ray
import torch
from torch_brain.datasets import DatasetIndex
from tqdm import tqdm

from core.dataset import BenchmarkRegime
from core.features import compute_isi_histogram
from core.utils.ray_map import imap_unordered
from ts3.models.base import Extractor
from ts3.ts3_dataset import IBLBrainWideBenchTS3


def _process_session_arrays(
    spike_timestamps, spike_units, unit_ids, n_bins, t_min, t_max, log_bins=True
) -> dict | None:
    """Compute ISI histograms for all units in a session. Safe to run in Ray worker."""
    try:
        n_units = len(unit_ids)
        hists = np.zeros((n_units, n_bins), dtype=np.float32)
        for i in range(n_units):
            hists[i] = compute_isi_histogram(
                spike_timestamps[spike_units == i], n_bins, t_min, t_max, log_bins
            )
        return {"hists": hists, "uids": unit_ids}
    except Exception as e:
        print(f"Failed: {e}")
        return None


@ray.remote
def _process_session_remote(**kwargs) -> dict | None:
    return _process_session_arrays(**kwargs)


class ISIExtractor(Extractor):
    def __init__(
        self,
        n_bins: int = 128,
        t_min: float = 1e-3,
        t_max: float = 3.0,
        log_bins: bool = True,
    ):
        self.n_bins = n_bins
        self.t_min = t_min
        self.t_max = t_max
        self.log_bins = log_bins

    @property
    def name(self) -> str:
        bins = "log" if self.log_bins else "lin"
        return f"isi_embs_{self.n_bins}{bins}_{self.t_min:g}_{self.t_max:g}"

    def encode(self, regime: BenchmarkRegime) -> tuple[torch.Tensor, np.ndarray]:
        dataset = IBLBrainWideBenchTS3(root=self.data_root, regime=regime)
        sampling_intervals = dataset.get_sampling_intervals()
        self.logger.info(f"{regime}: {len(dataset.recording_ids)} recordings")

        if not ray.is_initialized():
            ray.init(log_to_driver=False, ignore_reinit_error=True)

        def submit(rid):
            start, end = sampling_intervals[rid].start[0], sampling_intervals[rid].end[-1]
            recording = dataset[DatasetIndex(rid, start=start, end=end)]
            return _process_session_remote.remote(
                spike_timestamps=np.asarray(recording.spikes.timestamps),
                spike_units=np.asarray(recording.spikes.unit_index),
                unit_ids=np.asarray(recording.units.id).astype(str),
                n_bins=self.n_bins,
                t_min=self.t_min,
                t_max=self.t_max,
                log_bins=self.log_bins,
            )

        all_hists, all_uids = [], []
        results = imap_unordered(submit, dataset.recording_ids)
        # get_pbar sizes itself with len(), which imap_unordered's generator has not
        for result in tqdm(results, total=len(dataset.recording_ids), desc=f"[isi-{regime}]"):
            if result is not None:
                all_hists.append(result["hists"])
                all_uids.append(result["uids"])

        failed = len(dataset.recording_ids) - len(all_hists)
        if failed:
            self.logger.warn(
                f"{regime}: {failed} of {len(dataset.recording_ids)} recordings produced no "
                "histograms and are absent from the file (see the workers' 'Failed' lines)"
            )

        embs = np.concatenate(all_hists, axis=0)
        uids = np.concatenate(all_uids, axis=0)
        zero_rate = (embs.sum(axis=1) == 0).mean()
        self.logger.info(f"{regime}: {len(embs)} units | zero-histogram rate: {zero_rate:.3f}")

        return torch.from_numpy(embs).float(), uids
