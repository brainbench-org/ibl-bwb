"""Per-unit ridge autoregression, in closed form.

Maps each unit's last ``L`` observed bins to the forecast bins, predicting each future
bin separately so it captures the recent rate trend, not just its level. Reads observed
bins only.
"""

import logging

import torch

from ..base import LOG_EPS, StatBaseline

log = logging.getLogger(__name__)

AR_L_GRID = [3, 5, 10, 20, 45]  # autoregression window length (bins)
AR_LAMBDA_GRID = [0.1, 1.0, 10.0, 100.0, 1_000.0, 10_000.0]  # ridge strength


class RidgeAR(StatBaseline):
    """Per-unit ridge autoregression from the observed bins to the horizon.

    Args:
        window_length: autoregression window length (bins). ``None`` selects on
            val over ``AR_L_GRID``.
        ridge_lambda: ridge strength. ``None`` selects on val over
            ``AR_LAMBDA_GRID``.
    """

    TASK = "forecasting"
    NAME = "ridge_ar"

    def __init__(self, window_length: int | None = None, ridge_lambda: float | None = None):
        super().__init__()
        self._window_arg = window_length
        self._lambda_arg = ridge_lambda
        self.window_length: int = window_length if window_length is not None else 45
        self.ridge_lambda: float = ridge_lambda if ridge_lambda is not None else 1.0
        # Weights (N, L, forecast_ind) and intercept (N, forecast_ind).
        self.register_buffer("ar_w", torch.zeros(0), persistent=True)
        self.register_buffer("ar_b", torch.zeros(0), persistent=True)

    def fit(self, train_dataset, val_dataset):
        self._check_horizon()
        obs_end = self._obs_end
        ls = [self._window_arg] if self._window_arg is not None else AR_L_GRID
        lams = [self._lambda_arg] if self._lambda_arg is not None else AR_LAMBDA_GRID

        train = torch.from_numpy(self._windows(train_dataset))  # (n, T, N)
        val = torch.from_numpy(self._windows(val_dataset))  # (m, T, N)
        val_target = val[:, obs_end:, :]  # (m, forecast_ind, N)

        best = (-float("inf"), self.window_length, self.ridge_lambda, None, None)
        for L in ls:
            L_eff = min(L, obs_end)
            Xtr = train[:, obs_end - L_eff : obs_end, :]  # (n, L, N)
            Ytr = train[:, obs_end:, :]  # (n, forecast_ind, N)
            Xval = val[:, obs_end - L_eff : obs_end, :]  # (m, L, N)
            normal = self._ar_normal_equations(Xtr, Ytr)  # lambda-free, built once per L
            for lam in lams:
                W, b = self._ar_solve(*normal, lam)  # (N,L,F), (N,F)
                pred = self._ar_predict(Xval, W, b)  # (m, forecast_ind, N)
                score = self._score(pred.clamp_min(LOG_EPS).log(), val_target)
                if score > best[0]:
                    best = (score, L_eff, lam, W, b)

        self.val_score, self.window_length, self.ridge_lambda, W, b = best
        self.ar_w = W.contiguous()
        self.ar_b = b.contiguous()
        log.info(
            f"{self.NAME}: selected L={self.window_length} bins, lambda={self.ridge_lambda:.3g} "
            f"(per-unit ridge autoregression)"
        )

    def predict(self, spikes: torch.Tensor, **_) -> torch.Tensor:
        # Observed bins keep the mean rate; they are never scored.
        obs_end = spikes.shape[1] - self.forecast_ind
        x = spikes[:, obs_end - self.window_length : obs_end, :]  # (B, L, N)
        pred = self._ar_predict(x, self.ar_w.to(spikes.device), self.ar_b.to(spikes.device))
        out = self._log_mean_like(spikes)
        out[:, obs_end:, :] = pred.clamp_min(LOG_EPS).log()
        return out

    @staticmethod
    def _ar_normal_equations(X: torch.Tensor, Y: torch.Tensor):
        """Centered normal equations for the per-unit ridge.

        X (n, L, N), Y (n, F, N) -> gram (N, L, L), xty (N, L, F), Xm (L, N), Ym (F, N).
        Free of lambda, so the sweep builds them once per L instead of once per
        (L, lambda).
        """
        Xm = X.mean(dim=0)  # (L, N)
        Ym = Y.mean(dim=0)  # (F, N)
        Xc = X - Xm  # (n, L, N)
        Yc = Y - Ym  # (n, F, N)
        gram = torch.einsum("nli,nmi->ilm", Xc, Xc)  # (N, L, L)
        xty = torch.einsum("nli,nfi->ilf", Xc, Yc)  # (N, L, F)
        return gram, xty, Xm, Ym

    @staticmethod
    def _ar_solve(
        gram: torch.Tensor, xty: torch.Tensor, Xm: torch.Tensor, Ym: torch.Tensor, lam: float
    ):
        """Ridge solution at ``lam``, with intercept, via centering.

        Returns weights W (N, L, F) and intercept b (N, F).
        """
        L = gram.shape[-1]
        ridge = gram + lam * torch.eye(L, dtype=gram.dtype).unsqueeze(0)
        W = torch.linalg.solve(ridge, xty)  # (N, L, F)
        b = Ym.transpose(0, 1) - torch.einsum("il,ilf->if", Xm.transpose(0, 1), W)
        return W, b

    @staticmethod
    def _ar_predict(X: torch.Tensor, W: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        """X (m, L, N), W (N, L, F), b (N, F) -> predicted counts (m, F, N)."""
        return torch.einsum("bli,ilf->bfi", X, W) + b.transpose(0, 1).unsqueeze(0)
