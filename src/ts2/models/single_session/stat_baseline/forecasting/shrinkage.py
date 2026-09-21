"""Shrinkage blend of each unit's trial-local rate with its global mean.

The local rate is the observed-window mean, and
``rate = (1 - alpha) * mean_i + alpha * local_i`` is held flat over the horizon.
``alpha = 0`` recovers the mean rate, so the blend degrades safely when the local
signal is weak.
"""

import logging

import torch

from ..base import LOG_EPS, StatBaseline

log = logging.getLogger(__name__)

ALPHA_GRID = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]  # blend weight


class Shrinkage(StatBaseline):
    """Trial-local / global rate blend, held flat over the horizon.

    Args:
        alpha: blend weight on the local rate (0 = per-unit mean, 1 = pure local
            observed mean). ``None`` selects on val over ``ALPHA_GRID``.
    """

    TASK = "forecasting"
    NAME = "shrinkage"

    def __init__(self, alpha: float | None = None):
        super().__init__()
        self._alpha_arg = alpha
        self.alpha: float = alpha if alpha is not None else 0.5

    def fit(self, train_dataset, val_dataset):
        self._check_horizon()
        obs_end = self._obs_end
        alphas = [self._alpha_arg] if self._alpha_arg is not None else ALPHA_GRID

        val_counts = torch.from_numpy(self._windows(val_dataset))  # (B, T, N)
        local = val_counts[:, :obs_end, :].mean(dim=1)  # (B, N) observed-window mean
        target = val_counts[:, obs_end:, :]  # (B, forecast_ind, N)
        global_rate = self.log_mean.exp()[None, :]  # (1, N)

        best = (-float("inf"), self.alpha)
        for alpha in alphas:
            rate = (1.0 - alpha) * global_rate + alpha * local  # (B, N)
            pred = rate.clamp_min(LOG_EPS).log()[:, None, :].expand(-1, self.forecast_ind, -1)
            score = self._score(pred.contiguous(), target)
            if score > best[0]:
                best = (score, alpha)

        self.val_score, self.alpha = best
        log.info(
            f"{self.NAME}: selected alpha={self.alpha:.3g} "
            f"(local weight; alpha=0 is the per-unit mean)"
        )

    def predict(self, spikes: torch.Tensor, **_) -> torch.Tensor:
        B, T, N = spikes.shape
        obs_end = T - self.forecast_ind
        local = spikes[:, :obs_end, :].mean(dim=1)  # (B, N)
        global_rate = self.log_mean.to(spikes.device).exp()[None, :]  # (1, N)
        rate = (1.0 - self.alpha) * global_rate + self.alpha * local  # (B, N)
        log_rate = rate.clamp_min(LOG_EPS).log()[:, None, :]  # (B, 1, N)
        return log_rate.expand(B, T, N).contiguous()
