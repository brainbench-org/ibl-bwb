"""Trailing-window mean, held flat over the horizon.

Each unit's horizon is the average of its own last ``K`` observed bins. Reads observed
bins only, so val (full window) and test (future zeroed) give the identical prediction.
"""

import logging

import torch

from ..base import LOG_EPS, StatBaseline

log = logging.getLogger(__name__)

K_GRID = [1, 2, 3, 5, 8, 13, 21, 34, 45]  # trailing-window length (bins)


class TrailingMean(StatBaseline):
    """Trailing-window mean, held flat over the horizon.

    Args:
        k: trailing-window length (bins). ``None`` selects on val over ``K_GRID``.
    """

    TASK = "forecasting"
    NAME = "trailing_mean"

    def __init__(self, k: int | None = None):
        super().__init__()
        self._k_arg = k
        self.k: int = k if k is not None else 5

    def fit(self, train_dataset, val_dataset):
        self._check_horizon()
        obs_end = self._obs_end
        ks = [self._k_arg] if self._k_arg is not None else K_GRID

        val_counts = torch.from_numpy(self._windows(val_dataset))  # (B, T, N)
        target = val_counts[:, obs_end:, :]  # (B, forecast_ind, N)

        best = (-float("inf"), self.k)
        for k in ks:
            k_eff = min(k, obs_end)  # cannot look back past the observed portion
            tw_mean = val_counts[:, obs_end - k_eff : obs_end, :].mean(dim=1, keepdim=True)
            pred = tw_mean.clamp_min(LOG_EPS).log().expand(-1, self.forecast_ind, -1)
            score = self._score(pred.contiguous(), target)
            if score > best[0]:
                best = (score, k)

        self.val_score, self.k = best
        window_ms = self.k * self.bin_size * 1_000
        log.info(f"{self.NAME}: selected K={self.k} bins ({window_ms:.0f} ms trailing window)")

    def predict(self, spikes: torch.Tensor, **_) -> torch.Tensor:
        B, T, N = spikes.shape
        obs_end = T - self.forecast_ind
        k_eff = min(self.k, obs_end)
        tw_mean = spikes[:, obs_end - k_eff : obs_end, :].mean(dim=1, keepdim=True)
        log_rate = tw_mean.clamp_min(LOG_EPS).log()  # (B, 1, N)
        return log_rate.expand(B, T, N).contiguous()
