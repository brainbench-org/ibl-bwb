"""Tests for LOLCAT's class-balanced sampler: the mechanism, and the rule that moves it.

Three invariants. The epoch length is fixed at construction, since the LR schedule sizes
``T_max`` from it once. The bounds hold for every caller, not only :meth:`step`. And the
rule samples the hard classes more and the easy ones less, which is the half a bitwise
comparison against a restatement of the rule cannot catch.
"""

import pytest
import torch

from core.utils.util import global_mean_pool
from ts3.models.supervised.lolcat.sampler import (
    GROW_DIVISOR,
    OVERFIT_MARGIN,
    SHRINK_DIVISOR,
    LossFeedbackSampler,
)

LABELS = torch.tensor([0, 0, 0, 0, 1, 1, 2])  # counts 4, 2, 1

# two cells per class, class 2 the hardest to fit and class 0 the easiest
STEP_LABELS = torch.tensor([0, 0, 1, 1, 2, 2])
STEP_LOSS = torch.tensor([0.1, 0.1, 1.0, 1.0, 3.0, 3.0])


def _sampler(**kwargs):
    kwargs.setdefault("generator", torch.Generator().manual_seed(0))
    return LossFeedbackSampler(LABELS, **kwargs)


def test_an_integer_factor_repeats_every_index_that_many_times():
    counts = torch.bincount(_sampler(factor=3.0).indices)

    assert counts.tolist() == [3, 3, 3, 3, 3, 3, 3]


def test_a_fractional_factor_draws_the_remainder():
    """factor 1.5 on a class of 4 gives 4 copies plus 2 drawn at random."""
    indices = _sampler(factor=1.5).indices
    per_class = torch.bincount(LABELS[indices], minlength=3)

    assert per_class.tolist() == [6, 3, 1]  # 4*1.5, 2*1.5, floor(1*0.5) + 1


def test_factors_may_be_given_per_class():
    per_class = torch.bincount(LABELS[_sampler(factor=[1.0, 3.0, 5.0]).indices], minlength=3)

    assert per_class.tolist() == [4, 6, 5]


def test_a_factor_of_the_wrong_length_is_refused():
    with pytest.raises(ValueError, match="scalar or have 3 entries"):
        _sampler(factor=[1.0, 2.0])


def test_inverse_frequency_weighting_is_expressible():
    """The common static scheme: sample every class to the size of the largest."""
    counts = torch.bincount(LABELS, minlength=3)
    sampler = _sampler(factor=(counts.max() / counts).tolist(), bounds=(0.0, 100.0))
    per_class = torch.bincount(LABELS[sampler.indices], minlength=3)

    assert per_class.tolist() == [4, 4, 4]


def test_the_bounds_are_enforced_on_every_caller_not_only_on_step():
    sampler = _sampler(factor=5.0, bounds=(1.0, 3.0))
    sampler.set_factors(torch.tensor([0.1, 2.0, 99.0]))

    assert sampler.factors.tolist() == [1.0, 2.0, 3.0]


def test_the_initial_factors_are_bounded_too():
    assert _sampler(factor=500.0, bounds=(0.8, 100.0)).factors.tolist() == [100.0] * 3


def test_the_epoch_length_is_pinned_when_the_factors_move():
    """The LR schedule sizes T_max from len(sampler) once, so it must not drift."""
    sampler = _sampler(factor=5.0)
    before = len(sampler)

    sampler.set_factors(torch.tensor([1.0, 1.0, 1.0]))

    assert sampler.num_samples == before
    assert len(sampler) == len(sampler.indices)  # now the shorter of the two


def test_iterating_yields_one_epoch_of_indices():
    sampler = _sampler(factor=3.0)

    assert len(list(iter(sampler))) == len(sampler)


def _step(sampler, val_loss=None):
    sampler.step(
        train_loss=STEP_LOSS,
        train_labels=STEP_LABELS,
        val_loss=STEP_LOSS if val_loss is None else val_loss,
        val_labels=STEP_LABELS,
    )


def test_a_class_harder_than_average_is_sampled_more_and_an_easier_one_less():
    """What the rule is for: an epoch must not fill with the classes already fit."""
    sampler = _sampler(factor=5.0)

    _step(sampler)

    hard, easy = 2, 0
    assert sampler.factors[hard] == pytest.approx(5.0 / GROW_DIVISOR)
    assert sampler.factors[easy] == pytest.approx(5.0 / SHRINK_DIVISOR)


def test_divisors_of_one_hold_the_mix_where_the_factor_put_it():
    """How a run opts out of the feedback, now that both ends are configurable."""
    sampler = _sampler(factor=5.0, grow_divisor=1.0, shrink_divisor=1.0)

    _step(sampler)

    assert sampler.factors.tolist() == [5.0] * 3


def test_a_bigger_divisor_moves_a_class_further_in_one_step():
    """The step size is what decides whether the rule can reach the class imbalance."""
    slow = _sampler(factor=5.0, grow_divisor=0.99, shrink_divisor=1.01)
    fast = _sampler(factor=5.0, grow_divisor=0.96, shrink_divisor=1.04)

    _step(slow)
    _step(fast)

    hard = 2
    assert fast.factors[hard] > slow.factors[hard] > 5.0


def test_the_overfit_margin_decides_when_growth_stops():
    """At 0.0 any gap at all blocks growth, so the hard class stays where it started."""
    sampler = _sampler(factor=5.0, overfit_margin=0.0)

    _step(sampler)  # val equals train here, so the gap is exactly 0

    assert sampler.factors[2] == pytest.approx(5.0)


def test_a_hard_class_that_already_overfits_is_not_grown():
    """The val-train gap guards growth; it does not drive it."""
    sampler = _sampler(factor=5.0)
    std_train = torch.std(STEP_LOSS)
    # class 2 is still the hardest to fit, but its gap now exceeds the margin
    overfit = STEP_LOSS.clone()
    overfit[STEP_LABELS == 2] += OVERFIT_MARGIN * std_train + 1.0

    _step(sampler, val_loss=overfit)

    assert sampler.factors[2] == pytest.approx(5.0)


def test_a_class_missing_from_a_split_is_left_alone():
    """It pools to a loss of 0.0, which would otherwise read as a perfect score."""
    sampler = _sampler(factor=5.0)
    absent = 2
    labels = torch.tensor([0, 0, 1, 1])

    sampler.step(
        train_loss=torch.rand(4),
        train_labels=labels,
        val_loss=torch.rand(4),
        val_labels=labels,
    )

    assert sampler.factors[absent] == pytest.approx(5.0)
    assert not torch.equal(sampler.factors[:absent], torch.full((absent,), 5.0))


def test_a_degenerate_epoch_moves_nothing():
    """One cell per class leaves sigma undefined, and nan must not shrink every class."""
    sampler = _sampler(factor=5.0)
    labels = torch.tensor([0, 1, 2])

    sampler.step(
        train_loss=torch.tensor([1.0]),
        train_labels=torch.tensor([0]),
        val_loss=torch.rand(3),
        val_labels=labels,
    )

    assert sampler.factors.tolist() == [5.0] * 3


def _restated_rule(factors, train_loss, train_labels, val_loss, val_labels):
    """The rule, written out independently of the implementation under test."""
    factors = factors.clone()
    num_classes = factors.size(0)
    avg_train_loss = global_mean_pool(train_loss, train_labels, num_classes)
    global_std_train_loss, global_avg_train_loss = torch.std_mean(train_loss)
    avg_val_loss = global_mean_pool(val_loss, val_labels, num_classes)
    scored = (torch.bincount(val_labels, minlength=num_classes) > 0) & (
        torch.bincount(train_labels, minlength=num_classes) > 0
    )
    for i in range(num_classes):
        if not scored[i]:
            continue
        undertrained_score = (avg_train_loss[i] - global_avg_train_loss) / global_std_train_loss
        overfitting_score = avg_val_loss[i] - avg_train_loss[i]
        if undertrained_score > 0.0:
            if overfitting_score < 2.0 * global_std_train_loss:
                factors[i] = factors[i] / 0.96
        elif undertrained_score < 0.0:
            factors[i] = factors[i] / 1.04
    return torch.clip(factors, 0.8, 100.0)


def test_step_reproduces_the_rule_over_a_long_trajectory():
    """Bitwise, over long enough to catch a drifting constant or a clipped bound."""
    rng = torch.Generator().manual_seed(7)
    restated = torch.full((3,), 5.0)
    sampler = _sampler(factor=5.0)

    for _ in range(50):
        stats = {
            "train_loss": torch.rand(40, generator=rng),
            "train_labels": torch.randint(0, 3, (40,), generator=rng),
            "val_loss": torch.rand(20, generator=rng),
            "val_labels": torch.randint(0, 3, (20,), generator=rng),
        }
        restated = _restated_rule(restated, **stats)
        sampler.step(**stats)
        assert torch.equal(sampler.factors, restated)
