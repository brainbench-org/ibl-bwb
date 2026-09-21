from typing import Any

import torch
from torch import Tensor
from torchmetrics.metric import Metric


class PoissonD2Score(Metric):
    r"""Minimal implementation of Cohen pseudo-R^2 score for Poisson NLL loss.

    Supports 1D tensors (single output) and 2D tensors of shape (T, N) (multiple
    outputs). Assumes log-rates for the predictions.

    For 2D inputs, per-output R^2 scores are computed and the mean is returned.
    Outputs with zero total counts are excluded from the mean; if all outputs are
    degenerate the degenerate_value is returned.

    The number of outputs is inferred automatically from the last dimension of the
    first call to ``update()``, or can be provided explicitly via ``num_outputs``.

    Args:
        num_outputs: Number of outputs. If provided, states are pre-allocated eagerly;
            otherwise the size is inferred from the last dim of the first ``update()`` call.
        degenerate_value: The value to return when the metric is degenerate.
        eps: The epsilon value to use for numerical stability.
        validate_finite: Whether to check ``preds`` for NaN/inf on every ``update()``.
            Disable only when the caller already trusts ``preds`` is finite (e.g.
            scoring predictions written by a validated pipeline).

    The pseudo-R\ :sup:`2` score is computed as:

    .. math::

        1 - \frac{N_{pred} - N_{saturated}}{N_{null} - N_{saturated}}

    where :math:`N_{pred}` is the predicted Poisson NLL loss,
    :math:`N_{saturated}` is the saturated loss, and
    :math:`N_{null}` is the null (mean-rate) loss.

    """

    is_differentiable: bool = False
    higher_is_better: bool = True
    full_state_update: bool = False

    num_samples: Tensor
    sum_exp_rate: Tensor
    sum_interaction: Tensor
    sum_tgt_log_tgt: Tensor
    sum_target: Tensor

    def __init__(
        self,
        num_outputs: int | None = None,
        degenerate_value: Tensor | None = None,
        eps: float = 1e-9,
        validate_finite: bool = True,
        **kwargs: Any,
    ):
        super().__init__(**kwargs)

        self.eps = eps
        self.validate_finite = validate_finite
        self._num_outputs: int | None = None

        if degenerate_value is None:
            self.register_buffer("degenerate_value", torch.tensor(float("nan")))
        elif degenerate_value.numel() != 1:
            raise ValueError(
                f"degenerate_value must be a scalar, got {degenerate_value.numel()} elements"
            )
        else:
            self.register_buffer("degenerate_value", degenerate_value)

        self.add_state("num_samples", default=torch.tensor(0.0), dist_reduce_fx="sum")

        if num_outputs is not None:
            self._lazy_init(num_outputs)

    def _lazy_init(self, num_outputs: int):
        self._num_outputs = num_outputs
        self.add_state("sum_exp_rate", default=torch.zeros(num_outputs), dist_reduce_fx="sum")
        self.add_state("sum_interaction", default=torch.zeros(num_outputs), dist_reduce_fx="sum")
        self.add_state("sum_tgt_log_tgt", default=torch.zeros(num_outputs), dist_reduce_fx="sum")
        self.add_state("sum_target", default=torch.zeros(num_outputs), dist_reduce_fx="sum")

    def update(self, preds: Tensor, target: Tensor):
        """Accumulate predictions and targets into the metric state.

        Args:
            preds: Predicted log-rates of shape ``(T,)`` or ``(T, N)``.
            target: Target counts of shape ``(T,)`` or ``(T, N)``.
        """
        # Normalise to 2D (T, N)
        if preds.ndim == 1:
            preds = preds.unsqueeze(1)
        if target.ndim == 1:
            target = target.unsqueeze(1)

        if preds.ndim != 2:
            raise ValueError(f"preds must be 1D or 2D, got {preds.ndim}D")
        if target.ndim != 2:
            raise ValueError(f"target must be 1D or 2D, got {target.ndim}D")
        if preds.shape != target.shape:
            raise ValueError(
                f"preds and target must have the same shape, got {preds.shape} and {target.shape}"
            )

        if self._num_outputs is None:
            self._lazy_init(preds.shape[1])
            self.to(preds.device)
        elif preds.shape[1] != self._num_outputs:
            raise ValueError(
                f"preds has {preds.shape[1]} num_outputs but expected {self._num_outputs}"
            )

        if torch.any(target < 0):
            raise ValueError("target must be nonnegative for Poisson NLL")
        if self.validate_finite and not torch.isfinite(preds).all():
            raise ValueError("preds must not contain NaN or inf")

        target = target.float()
        self.sum_exp_rate += torch.sum(torch.exp(preds), dim=0)  # (N,)
        self.sum_interaction += torch.sum(preds * target, dim=0)  # (N,)
        self.sum_tgt_log_tgt += torch.sum(torch.xlogy(target, target), dim=0)  # (N,)
        self.sum_target += target.sum(dim=0)  # (N,)
        self.num_samples += preds.size(0)  # T

    def compute(self) -> Tensor:
        if self._num_outputs is None:
            return self.degenerate_value

        valid = self.sum_target > 0  # (N,) num_outputs with nonzero counts

        if not valid.any():
            return self.degenerate_value

        n_pred_loss = self.sum_exp_rate - self.sum_interaction  # (N,)
        n_sat_loss = self.sum_target - self.sum_tgt_log_tgt  # (N,)
        numerator = n_pred_loss - n_sat_loss

        null_rate = self.sum_target / self.num_samples  # (N,)
        denominator = self.sum_tgt_log_tgt - self.sum_target * torch.log(
            null_rate.clamp(min=self.eps)
        )  # (N,)

        valid = valid & (denominator.abs() >= self.eps)

        if not valid.any():
            return self.degenerate_value

        r2 = 1 - numerator[valid] / denominator[valid]  # (N_valid,)
        return r2.mean()
