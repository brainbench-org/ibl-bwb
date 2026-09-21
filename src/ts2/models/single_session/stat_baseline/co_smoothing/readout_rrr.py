"""Reduced-rank regression, the standard NLB baseline.

A rank-k ridge readout from the observed units to the held-out units at each timestep.
Unlike ``PopCoupling``'s rank-1 coupling this captures unit-specific, multi-dimensional
structure.
"""

import logging

import torch

from ._readout_base import RRRReadoutBase

log = logging.getLogger(__name__)


class RRRReadout(RRRReadoutBase):
    """Reduced-rank readout over binned counts.

    Args:
        rank: reduced rank k. ``None`` selects on val.
        ridge_lambda: ridge strength. ``None`` selects on val.
    """

    NAME = "readout_rrr"
    STANDARDIZE = False

    def fit(self, train_dataset, val_dataset):
        def features(dataset, obs_t, held_t):
            # cached per split, so both partitions share one walk
            counts = torch.from_numpy(self._windows(dataset).reshape(-1, self.N))
            return counts.index_select(1, obs_t), counts.index_select(1, held_t)

        score, obs_t, held_t = self._fit_two_stage(train_dataset, val_dataset, features)
        log.info(
            f"{self.NAME}: fit reduced-rank W ({obs_t.numel()} obs -> "
            f"{held_t.numel()} held-out), rank={self.rank}, "
            f"lambda={self.ridge_lambda:.3g}, val poisson_d2={score:.4f}"
        )

    def predict(self, spikes: torch.Tensor, **_) -> torch.Tensor:
        obs = self.rr_obs_idx.to(spikes.device)
        return self._scatter(spikes, spikes.index_select(-1, obs))  # (B, T, |O|)
