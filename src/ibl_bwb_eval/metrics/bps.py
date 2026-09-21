from typing import Any

import torch
from torch import Tensor
from torchmetrics.metric import Metric


class BPS(Metric):
    r"""Bits Per Spike metric for Poisson spiking data.

    Assumes log-rates for the predictions.

    The number of units is inferred automatically from the last dimension of the
    first call to ``update()``, or can be provided explicitly via ``num_outputs``.

    .. math::

        \frac{N_{null} - N_{pred}}{n_{sp} \cdot \log 2}

    Args:
        num_outputs: Number of units. If provided, states are pre-allocated eagerly;
            otherwise the size is inferred from the last dim of the first ``update()`` call.
        degenerate_value: The value to return when the metric is degenerate.
        eps: The epsilon value to use for numerical stability.
        validate_finite: Whether to check ``preds`` for NaN/inf on every ``update()``.
            Disable only when the caller already trusts ``preds`` is finite (e.g.
            scoring predictions written by a validated pipeline).

    References:
        :cite:`nlb`
    """

    is_differentiable: bool = False
    higher_is_better: bool = True
    full_state_update: bool = False

    num_samples: Tensor
    sum_target: Tensor
    sum_exp_rate: Tensor
    sum_interaction: Tensor

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

    def _lazy_init(self, num_outputs: int) -> None:
        self._num_outputs = num_outputs
        self.add_state("sum_target", default=torch.zeros(num_outputs), dist_reduce_fx="sum")
        self.add_state("sum_exp_rate", default=torch.zeros(num_outputs), dist_reduce_fx="sum")
        self.add_state("sum_interaction", default=torch.zeros(num_outputs), dist_reduce_fx="sum")

    def update(self, preds: Tensor, target: Tensor) -> None:
        """Accumulate predictions and targets into the metric state.

        Args:
            preds: Predicted log-rates of shape ``(T, N)``.
            target: Target spike counts of shape ``(T, N)``.
        """
        if preds.ndim != 2:
            raise ValueError(f"preds must be 2D (T, N), got {preds.ndim}D")
        if preds.shape != target.shape:
            raise ValueError(
                f"preds and target must have the same shape, got {preds.shape} and {target.shape}"
            )

        if self._num_outputs is None:
            self._lazy_init(preds.shape[1])
            self.to(preds.device)

        elif preds.shape[1] != self._num_outputs:
            raise ValueError(f"preds has {preds.shape[1]} units but expected {self._num_outputs}")

        if torch.any(target < 0):
            raise ValueError("target must be nonnegative for Poisson NLL")
        if self.validate_finite and not torch.isfinite(preds).all():
            raise ValueError("preds must not contain NaN or inf")

        target = target.float()
        self.sum_exp_rate += torch.sum(torch.exp(preds), dim=0)  # (N,)
        self.sum_interaction += torch.sum(preds * target, dim=0)  # (N,)
        self.sum_target += target.sum(dim=0)  # (N,)
        self.num_samples += preds.size(0)  # T

    def compute(self) -> Tensor:
        if self._num_outputs is None:
            return self.degenerate_value

        null_rate = self.sum_target / self.num_samples  # (N,)
        valid = null_rate >= self.eps  # (N,)

        if not valid.any():
            return self.degenerate_value

        total_spikes = self.sum_target[valid].sum()
        null_loss = torch.sum(self.sum_target[valid] * (1 - torch.log(null_rate[valid])))
        pred_loss = (self.sum_exp_rate - self.sum_interaction)[valid].sum()

        return (null_loss - pred_loss) / (total_spikes * torch.log(torch.tensor(2.0)))
