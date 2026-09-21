import numpy as np
import pytest
import torch
from scipy.special import gammaln
from torch import Tensor
from torch.nn import PoissonNLLLoss

from ibl_bwb_eval.metrics.poisson_d2 import PoissonD2Score

NUM_SEEDS = 100
ACCEPTABLE_TOLERANCE = 1e-4
ACCEPTABLE_RELATIVE_TOLERANCE = 1e-6


def ground_truth_poisson_nll_r2(preds: Tensor, target: Tensor) -> Tensor:
    pred_loss_fn = PoissonNLLLoss(
        log_input=True,
        full=True,
    )
    baseline_loss_fn = PoissonNLLLoss(
        log_input=False,
        full=True,
    )
    pred_loss = pred_loss_fn(preds, target)
    null_loss = baseline_loss_fn(torch.mean(target.float()), target)
    saturated_loss = baseline_loss_fn(target, target)
    return 1 - (pred_loss - saturated_loss) / (null_loss - saturated_loss)


def test_degenerate_value():
    # degenerate_value is parsed correctly
    assert PoissonD2Score().degenerate_value.isnan()
    assert PoissonD2Score(degenerate_value=torch.tensor(-1.0)).degenerate_value == -1.0
    # degenerate_value must be a scalar
    with pytest.raises(ValueError):
        PoissonD2Score(degenerate_value=torch.tensor([1.0, 2.0]))


def test_invalid_shapes_raise():
    # test shape validations
    metric = PoissonD2Score()
    with pytest.raises(ValueError):
        metric.update(torch.zeros(2, 3), torch.zeros(6, dtype=torch.long))
    with pytest.raises(ValueError):
        metric.update(torch.zeros(6), torch.zeros(2, 3, dtype=torch.long))
    with pytest.raises(ValueError):
        metric.update(torch.zeros(5), torch.zeros(6, dtype=torch.long))


def test_invalid_values_raise():
    # test value validations
    metric = PoissonD2Score()
    with pytest.raises(ValueError):
        metric.update(torch.zeros(3), torch.tensor([-1, 0, 1], dtype=torch.long))
    with pytest.raises(ValueError):
        metric.update(torch.tensor([float("nan"), 0.0, 0.0]), torch.ones(3, dtype=torch.long))
    with pytest.raises(ValueError):
        metric.update(torch.tensor([float("inf"), 0.0, 0.0]), torch.ones(3, dtype=torch.long))
    with pytest.raises(ValueError):
        metric.update(torch.tensor([float("-inf"), 0.0, 0.0]), torch.ones(3, dtype=torch.long))


def test_valid_edge_cases_do_not_raise():
    # test valid edge cases do not raise errors
    metric = PoissonD2Score()
    metric.update(torch.zeros(5), torch.zeros(5, dtype=torch.long))
    metric.update(torch.tensor([0.0]), torch.tensor([1], dtype=torch.long))
    metric.update(torch.zeros(5), torch.tensor([1_000, 999, 1_001, 998, 1_002], dtype=torch.long))


def test_degenerate_compute():
    # test all-zero targets
    for dv, check in [
        (None, lambda r: r.isnan()),
        (torch.tensor(-999.0), lambda r: r == -999.0),
    ]:
        metric = PoissonD2Score(degenerate_value=dv)
        metric.update(torch.zeros(5), torch.zeros(5, dtype=torch.long))
        assert check(metric.compute())

    # test uniform targets (denominator == 0)
    for dv, check in [
        (None, lambda r: r.isnan()),
        (torch.tensor(-999.0), lambda r: r == -999.0),
    ]:
        metric = PoissonD2Score(degenerate_value=dv)
        metric.update(torch.zeros(5), torch.ones(5, dtype=torch.long))
        assert check(metric.compute())


def test_xlogy_zero_times_log_zero():
    # test mixed zero/nonzero targets compute cleanly
    metric = PoissonD2Score()
    metric.update(torch.zeros(5), torch.tensor([0, 0, 1, 2, 3], dtype=torch.long))
    assert metric.sum_tgt_log_tgt.isfinite()
    assert not metric.compute().isnan()


def test_single_sample_compute_is_correct():
    # test that the metric computes the correct value for a single sample
    for seed in range(NUM_SEEDS):
        torch.manual_seed(seed)
        preds = torch.randn(100)
        target = torch.randint(0, 100, (100,))

        metric = PoissonD2Score()
        metric.update(preds, target)

        expected = ground_truth_poisson_nll_r2(preds, target)
        result = metric.compute()
        assert torch.isclose(result, expected), f"Result {result} != expected {expected}"


def test_multi_sample_compute_is_consistent():
    # test that the metric computes the same value whether using batch estimate or full
    for seed in range(NUM_SEEDS):
        torch.manual_seed(seed)

        # first, testing uniform batch sizes
        num_batches = 10
        batch_size = 100
        preds_batches = [torch.randn(batch_size) for _ in range(num_batches)]
        target_batches = [torch.randint(0, 100, (batch_size,)) for _ in range(num_batches)]

        preds_all = torch.cat(preds_batches, dim=0)
        target_all = torch.cat(target_batches, dim=0)

        metric = PoissonD2Score()
        metric.update(preds_all, target_all)
        expected = metric.compute()

        metric = PoissonD2Score()
        for preds, target in zip(preds_batches, target_batches, strict=True):
            metric.update(preds, target)
        result = metric.compute()

        assert torch.isclose(
            result,
            expected,
            rtol=ACCEPTABLE_RELATIVE_TOLERANCE,
            atol=ACCEPTABLE_TOLERANCE,
        ), f"Result {result} != expected {expected}"

        # then, testing non-uniform batch sizes
        num_batches = 10
        batch_sizes = torch.randint(1, 100, (num_batches,))
        preds_batches = [torch.randn(batch_size) for batch_size in batch_sizes]
        target_batches = [torch.randint(0, 100, (batch_size,)) for batch_size in batch_sizes]

        preds_all = torch.cat(preds_batches, dim=0)
        target_all = torch.cat(target_batches, dim=0)

        metric = PoissonD2Score()
        metric.update(preds_all, target_all)
        expected = metric.compute()

        metric = PoissonD2Score()
        for preds, target in zip(preds_batches, target_batches, strict=True):
            metric.update(preds, target)
        result = metric.compute()

        assert torch.isclose(
            result,
            expected,
            rtol=ACCEPTABLE_RELATIVE_TOLERANCE,
            atol=ACCEPTABLE_TOLERANCE,
        ), f"Result {result} != expected {expected}"


def test_2d_matches_per_output_mean():
    # 2D result should equal the mean of per-output 1D results
    for seed in range(NUM_SEEDS):
        torch.manual_seed(seed)
        T, N = 100, 5
        preds_2d = torch.randn(T, N)
        target_2d = torch.randint(0, 10, (T, N))

        metric_2d = PoissonD2Score()
        metric_2d.update(preds_2d, target_2d)
        result_2d = metric_2d.compute()

        per_output = []
        for n in range(N):
            m = PoissonD2Score()
            m.update(preds_2d[:, n], target_2d[:, n])
            per_output.append(m.compute())
        expected = torch.stack(per_output).mean()

        assert torch.isclose(result_2d, expected, atol=ACCEPTABLE_TOLERANCE), (
            f"2D result {result_2d} != mean of per-output {expected}"
        )


def test_2d_batched_consistent():
    # batched 2D updates should give the same result as a single update
    for seed in range(NUM_SEEDS):
        torch.manual_seed(seed)
        T, N, B = 10, 4, 5
        preds_batches = [torch.randn(T, N) for _ in range(B)]
        target_batches = [torch.randint(0, 10, (T, N)) for _ in range(B)]

        metric_all = PoissonD2Score()
        metric_all.update(torch.cat(preds_batches), torch.cat(target_batches))
        expected = metric_all.compute()

        metric_batched = PoissonD2Score()
        for p, t in zip(preds_batches, target_batches, strict=True):
            metric_batched.update(p, t)
        result = metric_batched.compute()

        assert torch.isclose(result, expected, atol=ACCEPTABLE_TOLERANCE), (
            f"Batched result {result} != single-update {expected}"
        )


def test_2d_degenerate_excludes_zero_spike_outputs():
    # outputs with all-zero counts should be silently excluded
    N = 4
    metric = PoissonD2Score()
    preds = torch.randn(50, N)
    target = torch.randint(0, 5, (50, N))
    target[:, 0] = 0  # output 0 has no counts

    metric.update(preds, target)
    result = metric.compute()

    # result should be finite (based on the 3 valid outputs)
    assert result.isfinite()


def test_2d_all_zero_counts_returns_degenerate():
    metric = PoissonD2Score()
    metric.update(torch.zeros(10, 3), torch.zeros(10, 3, dtype=torch.long))
    assert metric.compute().isnan()


def test_multi_sample_compute_is_correct():
    # note: this is redundant given the single sample correctness & multi-sample consistency tests pass,
    # but we will keep it as an independent test for coverage

    # test that the metric computes the correct value whether using batch estimate or full
    for seed in range(NUM_SEEDS):
        torch.manual_seed(seed)

        # first, testing uniform batch sizes
        num_batches = 10
        batch_size = 100
        preds_batches = [torch.randn(batch_size) for _ in range(num_batches)]
        target_batches = [torch.randint(0, 100, (batch_size,)) for _ in range(num_batches)]

        preds_all = torch.cat(preds_batches, dim=0)
        target_all = torch.cat(target_batches, dim=0)

        expected = ground_truth_poisson_nll_r2(preds_all, target_all)

        metric = PoissonD2Score()
        for preds, target in zip(preds_batches, target_batches, strict=True):
            metric.update(preds, target)
        result = metric.compute()

        assert torch.isclose(
            result,
            expected,
            rtol=ACCEPTABLE_RELATIVE_TOLERANCE,
            atol=ACCEPTABLE_TOLERANCE,
        ), f"Result {result} != expected {expected}"

        # then, testing non-uniform batch sizes
        num_batches = 10
        batch_sizes = torch.randint(1, 100, (num_batches,))
        preds_batches = [torch.randn(batch_size) for batch_size in batch_sizes]
        target_batches = [torch.randint(0, 100, (batch_size,)) for batch_size in batch_sizes]

        preds_all = torch.cat(preds_batches, dim=0)
        target_all = torch.cat(target_batches, dim=0)

        expected = ground_truth_poisson_nll_r2(preds_all, target_all)

        metric = PoissonD2Score()
        for preds, target in zip(preds_batches, target_batches, strict=True):
            metric.update(preds, target)
        result = metric.compute()

        assert torch.isclose(
            result,
            expected,
            rtol=ACCEPTABLE_RELATIVE_TOLERANCE,
            atol=ACCEPTABLE_TOLERANCE,
        ), f"Result {result} != expected {expected}"


# References DFE:
# Metric introduced in Xia et al. (2025). *Inpainting the Neural Picture: Inferring Unrecorded
# Brain Area Dynamics from Multi-Animal Datasets*. arXiv:2510.11924.
# https://arxiv.org/abs/2510.11924
# https://github.com/tinaxia2016/NeuroPaint/blob/main/src/utils/metric_utils.py#L117


def _numpy_poisson_fraction_deviance_explained(rates: np.ndarray, counts: np.ndarray) -> np.ndarray:
    """Reference numpy implementation of per-output Poisson fraction of deviance explained."""

    def _nll(r, s):
        r = r.copy()
        r[r < 1e-9] = 1e-9
        return np.sum(r - s * np.log(r) + gammaln(s + 1.0), axis=tuple(range(s.ndim - 1)))

    nll_model = _nll(rates, counts)
    null_rates = np.tile(
        np.nanmean(counts, axis=tuple(range(counts.ndim - 1)), keepdims=True),
        (*counts.shape[:-1], 1),
    )
    null_rates[null_rates < 1e-9] = 1e-9
    sat_rates = counts.copy()
    sat_rates[sat_rates < 1e-9] = 1e-9
    nll_null = _nll(null_rates, counts)
    nll_sat = _nll(sat_rates, counts)
    return 1.0 - (nll_model - nll_sat) / (nll_null - nll_sat + 1e-9)


def test_matches_numpy_poisson_fraction_deviance_explained():
    # PoissonD2Score (mean over outputs) should match the numpy reference implementation
    for seed in range(NUM_SEEDS):
        torch.manual_seed(seed)
        np.random.seed(seed)
        T, N = 100, 5
        # positive rates and nonzero counts to avoid degenerate outputs
        rates_np = np.abs(np.random.randn(T, N)) + 0.1
        counts_np = np.random.randint(1, 20, size=(T, N)).astype(float)

        expected = float(_numpy_poisson_fraction_deviance_explained(rates_np, counts_np).mean())

        log_rates = torch.tensor(np.log(rates_np), dtype=torch.float32)
        counts_t = torch.tensor(counts_np, dtype=torch.long)
        metric = PoissonD2Score()
        metric.update(log_rates, counts_t)
        result = metric.compute().item()

        assert abs(result - expected) < 1e-3, (
            f"seed={seed}: torch={result:.6f}, numpy={expected:.6f}"
        )
