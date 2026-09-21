import numpy as np
import pytest
import torch
from scipy.special import gammaln
from torch import Tensor
from torch.nn import PoissonNLLLoss

from ibl_bwb_eval.metrics.bps import BPS

NUM_SEEDS = 100
ACCEPTABLE_TOLERANCE = 1e-4
ACCEPTABLE_RELATIVE_TOLERANCE = 1e-6


# from https://github.com/neurallatents/nlb_tools/blob/main/nlb_tools/evaluation.py#L209
def _neg_log_likelihood(rates: np.ndarray, spikes: np.ndarray) -> float:
    """Poisson NLL: r - n*log(r) + log(n!), summed over all elements."""
    rates = rates.copy()
    if np.any(rates == 0):
        rates[rates == 0] = 1e-9
    result = rates - spikes * np.log(rates) + gammaln(spikes + 1.0)
    return float(np.sum(result))


def ground_truth_bps(log_rates: Tensor, spikes: Tensor) -> float:
    """Reference bits-per-spike using the original numpy implementation.

    Parameters
    ----------
    log_rates:
        Predicted log-rates, shape (T, N).
    spikes:
        Spike counts, shape (T, N).
    """
    rates = np.exp(log_rates.numpy())
    spikes_np = spikes.numpy().astype(float)

    nll_model = _neg_log_likelihood(rates, spikes_np)

    # null model: per-neuron mean rate tiled back to (T, N)
    # axis=tuple(range(ndim-1)) for 2D -> axis=(0,)
    null_rates = np.tile(
        np.nanmean(spikes_np, axis=tuple(range(spikes_np.ndim - 1)), keepdims=True),
        (*spikes_np.shape[:-1], 1),
    )
    nll_null = _neg_log_likelihood(null_rates, spikes_np)

    total_spikes = np.nansum(spikes_np)
    return (nll_null - nll_model) / total_spikes / np.log(2)


def ground_truth_bps_torch(log_rates: Tensor, spikes: Tensor) -> Tensor:
    """PyTorch-native bits-per-spike using PoissonNLLLoss with reduction='sum'."""
    spikes = spikes.float()
    pred_loss = PoissonNLLLoss(log_input=True, full=True, reduction="sum")(log_rates, spikes)
    null_rates = spikes.mean(dim=0, keepdim=True).expand_as(spikes)
    null_loss = PoissonNLLLoss(log_input=False, full=True, reduction="sum")(null_rates, spikes)
    return (null_loss - pred_loss) / (spikes.sum() * torch.log(torch.tensor(2.0)))


def test_degenerate_value():
    assert BPS().degenerate_value.isnan()
    assert BPS(degenerate_value=torch.tensor(-1.0)).degenerate_value == -1.0
    # degenerate_value must be a scalar
    with pytest.raises(ValueError):
        BPS(degenerate_value=torch.tensor([1.0, 2.0]))


def test_invalid_shapes_raise():
    with pytest.raises(ValueError):
        # 1D preds, must be 2D
        BPS().update(torch.zeros(4), torch.zeros(4, dtype=torch.long))
    with pytest.raises(ValueError):
        # wrong number of neurons after init
        metric = BPS(num_units=4)
        metric.update(torch.zeros(10, 3), torch.zeros(10, 3, dtype=torch.long))
    with pytest.raises(ValueError):
        # preds and target shape mismatch
        BPS().update(torch.zeros(10, 4), torch.zeros(5, 4, dtype=torch.long))


def test_invalid_values_raise():
    metric = BPS()
    with pytest.raises(ValueError):
        metric.update(
            torch.zeros(5, 4),
            torch.tensor([[-1, 0, 1, 2]] * 5, dtype=torch.long),
        )
    with pytest.raises(ValueError):
        preds = torch.zeros(5, 4)
        preds[0, 0] = float("nan")
        metric.update(preds, torch.ones(5, 4, dtype=torch.long))
    with pytest.raises(ValueError):
        preds = torch.zeros(5, 4)
        preds[0, 0] = float("inf")
        metric.update(preds, torch.ones(5, 4, dtype=torch.long))
    with pytest.raises(ValueError):
        preds = torch.zeros(5, 4)
        preds[0, 0] = float("-inf")
        metric.update(preds, torch.ones(5, 4, dtype=torch.long))


def test_valid_edge_cases_do_not_raise():
    metric = BPS()
    # zero log-rates, zero spikes
    metric.update(torch.zeros(5, 4), torch.zeros(5, 4, dtype=torch.long))
    # negative log-rates (valid: exp gives small positive rates)
    metric.update(torch.full((5, 4), -10.0), torch.ones(5, 4, dtype=torch.long))


def test_degenerate_compute():
    # all-zero targets -> total_spikes == 0
    for dv, check in [
        (None, lambda r: r.isnan()),
        (torch.tensor(-999.0), lambda r: r == -999.0),
    ]:
        metric = BPS(degenerate_value=dv)
        metric.update(torch.zeros(5, 4), torch.zeros(5, 4, dtype=torch.long))
        assert check(metric.compute())

    # one neuron with zero mean rate: silent neuron is excluded, result comes from active neuron only
    metric = BPS()
    target = torch.zeros(5, 2, dtype=torch.long)
    target[:, 0] = 1  # neuron 0 active, neuron 1 silent
    metric.update(torch.zeros(5, 2), target)
    assert torch.isclose(metric.compute(), torch.tensor(0.0))


def test_torch_ground_truth_matches_numpy():
    for seed in range(NUM_SEEDS):
        torch.manual_seed(seed)
        T, N = 50, 4
        log_rates = torch.randn(T, N)
        spikes = torch.randint(1, 10, (T, N))  # nonzero to avoid degenerate cases

        expected_np = ground_truth_bps(log_rates, spikes)
        result_torch = ground_truth_bps_torch(log_rates, spikes)
        assert torch.isclose(
            result_torch,
            torch.tensor(expected_np, dtype=result_torch.dtype),
            atol=ACCEPTABLE_TOLERANCE,
            rtol=ACCEPTABLE_RELATIVE_TOLERANCE,
        ), f"seed={seed}: torch={result_torch.item():.6f}, numpy={expected_np:.6f}"


def test_single_sample_compute_is_correct():
    for seed in range(NUM_SEEDS):
        torch.manual_seed(seed)
        T, N = 50, 4
        log_rates = torch.randn(T, N)
        spikes = torch.randint(0, 10, (T, N))

        metric = BPS()
        metric.update(log_rates, spikes)
        result = metric.compute()

        expected = ground_truth_bps(log_rates, spikes)
        assert torch.isclose(
            result,
            torch.tensor(expected, dtype=result.dtype),
            atol=ACCEPTABLE_TOLERANCE,
            rtol=ACCEPTABLE_RELATIVE_TOLERANCE,
        ), f"seed={seed}: result={result.item():.6f}, expected={expected:.6f}"


def test_multi_sample_compute_is_consistent():
    for seed in range(NUM_SEEDS):
        torch.manual_seed(seed)
        N = 4

        # uniform batch sizes
        num_batches, batch_size = 10, 50
        log_rate_batches = [torch.randn(batch_size, N) for _ in range(num_batches)]
        spike_batches = [torch.randint(0, 10, (batch_size, N)) for _ in range(num_batches)]

        log_rates_all = torch.cat(log_rate_batches, dim=0)
        spikes_all = torch.cat(spike_batches, dim=0)

        metric_full = BPS()
        metric_full.update(log_rates_all, spikes_all)
        expected = metric_full.compute()

        metric_batched = BPS()
        for lr, sp in zip(log_rate_batches, spike_batches, strict=True):
            metric_batched.update(lr, sp)
        result = metric_batched.compute()

        assert torch.isclose(
            result,
            expected,
            atol=ACCEPTABLE_TOLERANCE,
            rtol=ACCEPTABLE_RELATIVE_TOLERANCE,
        ), f"seed={seed}: result={result.item():.6f}, expected={expected.item():.6f}"

        # non-uniform batch sizes
        batch_sizes = torch.randint(1, 50, (num_batches,))
        log_rate_batches = [torch.randn(int(bs), N) for bs in batch_sizes]
        spike_batches = [torch.randint(0, 10, (int(bs), N)) for bs in batch_sizes]

        log_rates_all = torch.cat(log_rate_batches, dim=0)
        spikes_all = torch.cat(spike_batches, dim=0)

        metric_full = BPS()
        metric_full.update(log_rates_all, spikes_all)
        expected = metric_full.compute()

        metric_batched = BPS()
        for lr, sp in zip(log_rate_batches, spike_batches, strict=True):
            metric_batched.update(lr, sp)
        result = metric_batched.compute()

        assert torch.isclose(
            result,
            expected,
            atol=ACCEPTABLE_TOLERANCE,
            rtol=ACCEPTABLE_RELATIVE_TOLERANCE,
        ), f"seed={seed}: result={result.item():.6f}, expected={expected.item():.6f}"


def test_multi_sample_compute_is_correct():
    # redundant given single-sample correctness + multi-sample consistency,
    # but kept as an independent end-to-end check
    for seed in range(NUM_SEEDS):
        torch.manual_seed(seed)
        N = 4

        # uniform batch sizes
        num_batches, batch_size = 10, 50
        log_rate_batches = [torch.randn(batch_size, N) for _ in range(num_batches)]
        spike_batches = [torch.randint(0, 10, (batch_size, N)) for _ in range(num_batches)]

        log_rates_all = torch.cat(log_rate_batches, dim=0)
        spikes_all = torch.cat(spike_batches, dim=0)
        expected = ground_truth_bps(log_rates_all, spikes_all)

        metric = BPS()
        for lr, sp in zip(log_rate_batches, spike_batches, strict=True):
            metric.update(lr, sp)
        result = metric.compute()

        assert torch.isclose(
            result,
            torch.tensor(expected, dtype=result.dtype),
            atol=ACCEPTABLE_TOLERANCE,
            rtol=ACCEPTABLE_RELATIVE_TOLERANCE,
        ), f"seed={seed}: result={result.item():.6f}, expected={expected:.6f}"

        # non-uniform batch sizes
        batch_sizes = torch.randint(1, 50, (num_batches,))
        log_rate_batches = [torch.randn(int(bs), N) for bs in batch_sizes]
        spike_batches = [torch.randint(0, 10, (int(bs), N)) for bs in batch_sizes]

        log_rates_all = torch.cat(log_rate_batches, dim=0)
        spikes_all = torch.cat(spike_batches, dim=0)
        expected = ground_truth_bps(log_rates_all, spikes_all)

        metric = BPS()
        for lr, sp in zip(log_rate_batches, spike_batches, strict=True):
            metric.update(lr, sp)
        result = metric.compute()

        assert torch.isclose(
            result,
            torch.tensor(expected, dtype=result.dtype),
            atol=ACCEPTABLE_TOLERANCE,
            rtol=ACCEPTABLE_RELATIVE_TOLERANCE,
        ), f"seed={seed}: result={result.item():.6f}, expected={expected:.6f}"
