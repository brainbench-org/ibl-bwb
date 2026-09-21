"""ISI-augmented reduced-rank readout, the same regression as ``RRRReadout``.

Each observed unit also contributes its per-bin time-since-last-spike and most-recent
ISI, so the regressors carry the sub-bin timing that binning discards. Features are
standardized, since they mix counts with seconds.
"""

import logging

import numpy as np
import torch
from torch_brain.data import Data
from torch_brain.samplers import SequentialFixedWindowSampler

from ts2.ts2_dataset import IBLBrainWideBenchTS2

from ._readout_base import RRRReadoutBase

log = logging.getLogger(__name__)


class RRRReadoutWithISI(RRRReadoutBase):
    """Reduced-rank readout over [counts, time-since-last-spike, last ISI].

    Args:
        rank: reduced rank k. ``None`` selects on val.
        ridge_lambda: ridge strength. ``None`` selects on val.
    """

    NAME = "readout_isi"
    STANDARDIZE = True

    def __init__(self, rank: int | None = None, ridge_lambda: float | None = None):
        super().__init__(rank, ridge_lambda)
        self._isi_cache: dict[str, tuple] = {}

    def input_fn(self, data: Data) -> dict:
        tsls, last_isi = self._isi_features(data)  # (T, N) each
        return {
            "model_inputs": {
                "spikes": self._bin(data),
                "tsls": tsls,
                "last_isi": last_isi,
            }
        }

    def fit(self, train_dataset, val_dataset):
        def features(dataset, obs_t, held_t):
            # (n, T, N) each -> regressors (M, 3|O|) and targets (M, |H|).
            counts, tsls, isi = self._readout_features(dataset)
            C = torch.from_numpy(counts.reshape(-1, self.N))
            Tt = torch.from_numpy(tsls.reshape(-1, self.N))
            Ii = torch.from_numpy(isi.reshape(-1, self.N))
            X = torch.cat(
                [
                    C.index_select(1, obs_t),
                    Tt.index_select(1, obs_t),
                    Ii.index_select(1, obs_t),
                ],
                dim=1,
            )
            return X, C.index_select(1, held_t)

        score, obs_t, held_t = self._fit_two_stage(train_dataset, val_dataset, features)
        self._isi_cache.clear()
        log.info(
            f"{self.NAME}: fit ISI-augmented reduced-rank W "
            f"({3 * obs_t.numel()} features -> {held_t.numel()} held-out), "
            f"rank={self.rank}, lambda={self.ridge_lambda:.3g}, val poisson_d2={score:.4f}"
        )

    def predict(self, spikes, tsls=None, last_isi=None, **_) -> torch.Tensor:
        assert tsls is not None and last_isi is not None, (
            f"{self.NAME} forward requires the 'tsls' and 'last_isi' model inputs"
        )
        obs = self.rr_obs_idx.to(spikes.device)
        xo = torch.cat(
            [
                spikes.index_select(-1, obs),
                tsls.index_select(-1, obs),
                last_isi.index_select(-1, obs),
            ],
            dim=-1,
        )  # (B, T, 3|O|)
        return self._scatter(spikes, xo)

    # ------------------------------------------------------------------
    # ISI features
    # ------------------------------------------------------------------
    def _isi_features(self, data: Data):
        """Time-since-last-spike and most-recent ISI per unit, (T, N) each in seconds.

        Evaluated at each bin's end using only spikes up to then.
        """
        spikes = data.spikes
        ts = np.asarray(spikes.timestamps)
        ui = np.asarray(spikes.unit_index)
        start = float(spikes.domain.start[0])
        tau = start + (np.arange(self.T) + 1) * self.bin_size  # bin-end times (T,)
        dur = self.T * self.bin_size
        tsls = np.empty((self.T, self.N), dtype=np.float32)
        isi = np.zeros((self.T, self.N), dtype=np.float32)
        for u in range(self.N):
            su = np.sort(ts[ui == u])
            if su.size == 0:
                tsls[:, u] = tau - start  # elapsed since window start
                continue
            idx = np.searchsorted(su, tau, side="right") - 1  # last spike <= tau
            has = idx >= 0
            last = np.where(has, su[np.clip(idx, 0, su.size - 1)], start)
            tsls[:, u] = np.where(has, tau - last, tau - start)
            has2 = idx >= 1
            prev = np.where(has2, su[np.clip(idx - 1, 0, su.size - 1)], 0.0)
            isi[:, u] = np.where(has2, last - prev, 0.0)
        np.clip(tsls, 0.0, dur, out=tsls)
        np.clip(isi, 0.0, dur, out=isi)
        return tsls, isi

    def _readout_features(self, dataset: IBLBrainWideBenchTS2):
        """Cached :meth:`_collect_readout_features`, keyed by split.

        The ISI walk dominates the fit and both partitions need the same one.
        """
        if dataset.split not in self._isi_cache:
            self._isi_cache[dataset.split] = self._collect_readout_features(dataset)
        return self._isi_cache[dataset.split]

    def _collect_readout_features(self, dataset: IBLBrainWideBenchTS2):
        """Per window counts, tsls, isi; each (n_windows, T, N)."""
        sampler = SequentialFixedWindowSampler(
            sampling_intervals=dataset.get_sampling_intervals(),
            window_length=dataset.context_window,
            step=dataset.context_window,
        )
        counts_l, tsls_l, isi_l = [], [], []
        for index in sampler:
            data = dataset.dataset_transform(
                dataset.get_recording(index.recording_id).slice(index.start, index.end)
            )
            counts = self._bin(data)
            if counts.shape[0] != self.T:
                continue
            tsls, isi = self._isi_features(data)
            counts_l.append(counts)
            tsls_l.append(tsls)
            isi_l.append(isi)
        assert counts_l, f"no complete windows found while fitting {self.NAME}"
        return np.stack(counts_l), np.stack(tsls_l), np.stack(isi_l)
