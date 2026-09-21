"""Rank-1 population coupling: ``r_i(t) = mean_i * p(t) ** gamma``.

``p(t)`` is a smoothed, mean-1-normalized population trace. Collapses to the mean rate
at ``gamma = 0``.
"""

import hashlib
import logging

import numpy as np
import torch
import torch.nn.functional as F
from scipy.ndimage import gaussian_filter1d

from ts2.ts2_dataset import IBLBrainWideBenchTS2

from ..base import LOG_EPS, StatBaseline

log = logging.getLogger(__name__)

SIGMA_GRID = [1.0, 1.5, 2.0, 2.5]  # Gaussian smoothing width
GAMMA_GRID = [0.5, 0.75, 1.0, 1.25, 1.5, 2.0]  # coupling exponent
FOLD_SALT = "pop_coupling_fold"  # unit-fold assignment, see _unit_folds


class PopCoupling(StatBaseline):
    """Population coupling.

    Args:
        sigma: Gaussian smoothing width (bins). ``None`` selects on val.
        gamma: coupling exponent. ``None`` selects on val.
    """

    TASK = "co_smoothing"
    NAME = "pop_coupling"

    def __init__(self, sigma: float | None = None, gamma: float | None = None):
        super().__init__()
        self._sigma_arg = sigma
        self._gamma_arg = gamma
        self.sigma: float = sigma if sigma is not None else 1.5
        self.gamma: float = gamma if gamma is not None else 1.0
        self.register_buffer("smooth_kernel", self._gaussian_kernel(self.sigma), persistent=True)

    def fit(self, train_dataset, val_dataset: IBLBrainWideBenchTS2):
        """Select sigma and gamma on val.

        Val feeds the full population, so a population sum including unit i would
        leak its own target into its prediction. Instead of leaving out only unit i,
        the trace leaves out a whole fold the size of the test hold-out, so selection
        reads the population the model reads at test; the folds tile the units, so
        every unit is still scored.
        """
        sigmas = [self._sigma_arg] if self._sigma_arg is not None else SIGMA_GRID
        gammas = [self._gamma_arg] if self._gamma_arg is not None else GAMMA_GRID

        n_folds = min(max(2, round(1 / val_dataset.CO_SMOOTHING_HELD_OUT_RATIO)), self.N)
        folds = self._unit_folds(val_dataset, n_folds)  # (N,)
        val_windows = self._windows(val_dataset)  # (n_windows, T, N)
        val_counts = torch.from_numpy(val_windows)  # (B, T, N)
        log_mean_t = self.log_mean  # (N,)

        best = (-float("inf"), self.sigma, self.gamma)
        for sigma in sigmas:
            p = self._population_multiplier_np(val_windows, sigma, folds, n_folds)
            log_p = np.log(np.clip(p, LOG_EPS, None)).astype(np.float32)
            log_p_t = torch.from_numpy(log_p)  # (B, T, N)
            for gamma in gammas:
                pred = log_mean_t.view(1, 1, -1) + gamma * log_p_t  # (B, T, N)
                score = self._score(pred, val_counts)
                if score > best[0]:
                    best = (score, sigma, gamma)

        self.val_score, self.sigma, self.gamma = best
        self.smooth_kernel = self._gaussian_kernel(self.sigma)
        log.info(f"{self.NAME}: selected sigma={self.sigma:.3g}, gamma={self.gamma:.3g}")

    def predict(self, spikes: torch.Tensor, **_) -> torch.Tensor:
        # At test the held-out units are already removed from the input, so the
        # population sum is over the observed units only (no self-term).
        if self.gamma == 0.0:
            return self._log_mean_like(spikes)
        return self._log_mean_view(spikes) + self.gamma * self._log_pop_multiplier(spikes)

    def _log_pop_multiplier(self, spikes: torch.Tensor) -> torch.Tensor:
        """Smoothed, per-window mean-1-normalized log population trace, (B, T, 1)."""
        pop = spikes.sum(dim=-1)  # (B, T)
        pop_smooth = self._smooth_torch(pop)  # (B, T)
        p = pop_smooth / pop_smooth.mean(dim=1, keepdim=True).clamp_min(LOG_EPS)
        return p.clamp_min(LOG_EPS).log().unsqueeze(-1)  # (B, T, 1)

    def _smooth_torch(self, pop: torch.Tensor) -> torch.Tensor:
        """Gaussian smooth over time, edge='nearest' (replicate padding)."""
        kernel = self.smooth_kernel.to(pop.device, pop.dtype)  # (1, 1, K)
        radius = (kernel.shape[-1] - 1) // 2
        x = F.pad(pop.unsqueeze(1), (radius, radius), mode="replicate")  # (B, 1, T)
        return F.conv1d(x, kernel).squeeze(1)  # (B, T)

    @staticmethod
    def _gaussian_kernel(sigma: float, truncate: float = 4.0) -> torch.Tensor:
        """Normalized Gaussian as a (1, 1, K) conv weight.

        Follows scipy's radius convention so the torch forward and the numpy val sweep
        agree.
        """
        radius = max(1, int(truncate * sigma + 0.5))
        x = np.arange(-radius, radius + 1, dtype=np.float64)
        w = np.exp(-0.5 * (x / sigma) ** 2)
        w /= w.sum()
        return torch.from_numpy(w.astype(np.float32)).view(1, 1, -1)

    @staticmethod
    def _unit_folds(dataset: IBLBrainWideBenchTS2, n_folds: int) -> np.ndarray:
        """Fold index per unit (N,), in unit-table order.

        Keyed on ``sha256(salt:unit_id)``, so it survives a reordered unit table.
        """
        ids = dataset.get_recording(dataset.recording_id).units.id
        digests = np.array(
            [
                hashlib.sha256(
                    f"{FOLD_SALT}:{u.decode() if isinstance(u, bytes) else u}".encode()
                ).hexdigest()
                for u in ids
            ]
        )
        folds = np.empty(len(ids), dtype=np.int64)
        for f, idx in enumerate(np.array_split(np.argsort(digests), n_folds)):
            folds[idx] = f
        return folds

    @staticmethod
    def _population_multiplier_np(
        windows: np.ndarray, sigma: float, folds: np.ndarray, n_folds: int
    ) -> np.ndarray:
        """Population multiplier (n_windows, T, N).

        Entry (b, t, i) is the smoothed population at t over the units outside i's fold,
        mean-1 over time. Smoothing is linear, so the smoothed total is the sum of the
        per-unit smoothed traces and a fold's trace is that total minus the fold's own
        units.
        """
        unit_smooth = gaussian_filter1d(windows, sigma=sigma, axis=1, mode="nearest")
        pop_smooth = unit_smooth.sum(axis=-1, keepdims=True)  # (B, T, 1)
        onehot = np.zeros((folds.size, n_folds), dtype=unit_smooth.dtype)
        onehot[np.arange(folds.size), folds] = 1.0
        trace = pop_smooth - unit_smooth @ onehot  # (B, T, n_folds)
        denom = np.clip(trace.mean(axis=1, keepdims=True), LOG_EPS, None)  # (B, 1, n_folds)
        return (trace / denom)[:, :, folds]  # (B, T, N)
