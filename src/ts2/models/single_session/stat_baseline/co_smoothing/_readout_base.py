"""Shared machinery for the observed-to-held-out reduced-rank readouts.

Both methods map the observed units ``O`` to the held-out units ``H`` at each
timestep with a rank-constrained ridge, differing only in their regressors:
``RRRReadout`` uses binned counts, ``RRRReadoutWithISI`` augments them with spike timing. H is
the fixed ``is_held_out_test`` unit mask (the units scored at test),
and since H and O are disjoint, predicting H from O is leakage-free.

Hyperparameters are selected on the val hold-out draw, then the readout is refit for
the test one, so the scored units take no part in selection.
"""

import numpy as np
import torch

from ts2.ts2_dataset import IBLBrainWideBenchTS2

from ..base import LOG_EPS, StatBaseline

# Reduced rank, capped at min(|O|, |H|). The ceiling of 16 sits below |H| on a
# typical session (10% of a few hundred units), so the sweep never reaches the
# rank-|H| point at which this becomes the full ridge.
RANK_GRID = [1, 2, 3, 4, 5, 8, 12, 16]
READOUT_LAMBDA_GRID = [1.0, 10.0, 100.0, 1e3, 1e4, 1e5, 1e6, 1e7]  # ridge strength


class RRRReadoutBase(StatBaseline):
    """Base for the observed-to-held-out reduced-rank readouts.

    Args:
        rank: reduced rank k. ``None`` selects over ``RANK_GRID``.
        ridge_lambda: ridge strength. ``None`` selects over ``READOUT_LAMBDA_GRID``.
    """

    TASK = "co_smoothing"
    # Standardize regressors before the solve. Needed when features mix units
    # (counts with seconds), unnecessary for homogeneous counts.
    STANDARDIZE: bool = False
    HELD_OUT_ATTR: str = "ts2_co_smoothing_is_held_out_test"
    SELECT_ON_ATTR: str = "ts2_co_smoothing_is_held_out_val"

    def __init__(self, rank: int | None = None, ridge_lambda: float | None = None):
        super().__init__()
        self._rank_arg = rank
        self._lambda_arg = ridge_lambda
        self.rank: int = rank if rank is not None else 5
        self.ridge_lambda: float = ridge_lambda if ridge_lambda is not None else 100.0
        # Weights (P, |H|), intercept (|H|,), observed / held-out column indices.
        self.register_buffer("rr_w", torch.zeros(0), persistent=True)
        self.register_buffer("rr_b", torch.zeros(0), persistent=True)
        self.register_buffer("rr_obs_idx", torch.zeros(0, dtype=torch.long), persistent=True)
        self.register_buffer("rr_held_idx", torch.zeros(0, dtype=torch.long), persistent=True)

    def _held_out_mask(self, dataset: IBLBrainWideBenchTS2, attr: str) -> np.ndarray:
        """Held-out unit mask (N,).

        Off the recording, not a sliced item: ``dataset_transform`` keeps at most the
        item's own split's mask.
        """
        rec = dataset.get_recording(dataset.recording_id)
        assert hasattr(rec.units, attr), (
            f"{attr} missing; {self.NAME} needs data with split val/test hold-outs"
        )
        return np.asarray(getattr(rec.units, attr), dtype=bool)

    def _split_units(self, dataset: IBLBrainWideBenchTS2, attr: str | None = None):
        """Observed / held-out column indices under ``attr``'s mask."""
        held = self._held_out_mask(dataset, attr or self.HELD_OUT_ATTR)
        obs_idx, held_idx = np.where(~held)[0], np.where(held)[0]
        assert held_idx.size and obs_idx.size, (
            f"co_smoothing needs both observed and held-out units for {self.NAME}"
        )
        return torch.from_numpy(obs_idx), torch.from_numpy(held_idx)

    def _grids(self):
        """(ranks, lambdas) to sweep, honouring any pinned constructor values."""
        ks = [self._rank_arg] if self._rank_arg is not None else RANK_GRID
        lams = [self._lambda_arg] if self._lambda_arg is not None else READOUT_LAMBDA_GRID
        return ks, lams

    def _store(self, rank, ridge_lambda, w, b, obs_t, held_t):
        self.rank = rank
        self.ridge_lambda = ridge_lambda
        self.rr_w = w.contiguous()
        self.rr_b = b.contiguous()
        self.rr_obs_idx = obs_t.long()
        self.rr_held_idx = held_t.long()

    def _fit_reduced_rank(self, Xtr, Ytr, ks, lams, standardize: bool, Xval=None, Yval=None):
        """Reduced-rank ridge from Xtr (M, P) to Ytr (M, |H|).

        Each lambda's full ridge solution is projected onto its top-k output
        dimensions, ``W_rrr = W_ridge V_k V_k'``, where ``V_k`` holds the top-k
        right singular vectors of the fitted responses (via the |H| x |H| output
        covariance, whose top eigenvectors are those RSVs). Rank-|H| is the full
        ridge. Returns (score, rank, lambda, W_eff, b_eff), with standardization
        folded in so the readout stays ``X @ W_eff + b_eff``.

        With (Xval, Yval) the grid is swept and the best poisson_d2 wins; without
        them the grid must be a single point, fitted unscored (score ``nan``).
        """
        select = Xval is not None and Yval is not None
        assert select or (len(ks) == 1 and len(lams) == 1), (
            f"{self.NAME}: no val data, so k and lambda must be pinned"
        )
        Xm, Ym = Xtr.mean(dim=0), Ytr.mean(dim=0)
        Xstd = Xtr.std(dim=0).clamp_min(1e-6) if standardize else torch.ones_like(Xm)
        Xs, Yc = (Xtr - Xm) / Xstd, Ytr - Ym
        gram = Xs.transpose(0, 1) @ Xs  # (P, P)
        xty = Xs.transpose(0, 1) @ Yc  # (P, |H|)
        n_feat = Xtr.shape[1]
        rank_max = min(n_feat, Ytr.shape[1])
        eye = torch.eye(n_feat, dtype=Xtr.dtype)

        best = (-float("inf"), ks[0], lams[0], None, None)
        for lam in lams:
            w_ridge = torch.linalg.solve(gram + lam * eye, xty)  # (P, |H|)
            ycov = w_ridge.transpose(0, 1) @ gram @ w_ridge  # (|H|, |H|)
            v_desc = torch.flip(torch.linalg.eigh(ycov).eigenvectors, dims=(-1,))
            for k in ks:
                kk = min(k, rank_max, v_desc.shape[1])
                vk = v_desc[:, :kk]
                w = w_ridge @ (vk @ vk.transpose(0, 1))  # (P, |H|), rank <= kk
                w_eff = w / Xstd.unsqueeze(1)  # fold input standardization
                b_eff = Ym - Xm @ w_eff
                if not select:
                    return (float("nan"), kk, lam, w_eff, b_eff)
                pred = (Xval @ w_eff + b_eff).clamp_min(LOG_EPS).log()
                score = self._score(pred.unsqueeze(1), Yval.unsqueeze(1))
                if score > best[0]:
                    best = (score, kk, lam, w_eff, b_eff)
        return best

    def _fit_two_stage(self, train_dataset, val_dataset, features):
        """Select (k, lambda) on the val draw, refit for the test draw, store it.

        ``features(dataset, obs, held)`` yields regressors (M, P) and targets
        (M, |H|). Records the selection score as ``val_score`` and returns it with
        the stored partition.
        """
        ks, lams = self._grids()
        obs_v, held_v = self._split_units(train_dataset, self.SELECT_ON_ATTR)
        score, rank, lam, _, _ = self._fit_reduced_rank(
            *features(train_dataset, obs_v, held_v),
            ks,
            lams,
            self.STANDARDIZE,
            *features(val_dataset, obs_v, held_v),
        )
        self.val_score = score

        obs_t, held_t = self._split_units(train_dataset)
        _, rank, lam, w, b = self._fit_reduced_rank(
            *features(train_dataset, obs_t, held_t), [rank], [lam], self.STANDARDIZE
        )
        self._store(rank, lam, w, b, obs_t, held_t)
        return score, obs_t, held_t

    def _scatter(self, spikes: torch.Tensor, xo: torch.Tensor) -> torch.Tensor:
        """Map observed regressors ``xo`` (B, T, P) onto the held-out units.

        Units outside H keep the mean rate and are never scored.
        """
        out = self._log_mean_like(spikes)
        held = self.rr_held_idx.to(spikes.device)
        pred = torch.einsum("btp,ph->bth", xo, self.rr_w.to(spikes.device))
        pred = pred + self.rr_b.to(spikes.device)  # (B, T, |H|)
        out[:, :, held] = pred.clamp_min(LOG_EPS).log()
        return out
