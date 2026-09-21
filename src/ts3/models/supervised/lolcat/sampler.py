"""LOLCAT's class-balanced sampler, and the loss-feedback rule that moves it.

Classes harder to fit than average are sampled more and classes already fit are sampled
less, so an epoch does not fill up with the largest region. TS3 scores macro F1 over ten
regions of very unequal size.
"""

from collections.abc import Iterator, Sequence

import torch
from torch.utils.data.sampler import Sampler

from core.utils.util import global_mean_pool

# a factor moves about 4% per val epoch, enough to reach the region imbalance (19x) within
# a run; two sigmas of val-train gap is what counts as overfitting
GROW_DIVISOR = 0.96
SHRINK_DIVISOR = 1.04
OVERFIT_MARGIN = 2.0


class LossFeedbackSampler(Sampler[int]):
    r"""Samples elements randomly from a given list of indices, without replacement.

    Each class contributes ``factor[c]`` times its own index count to an epoch: the
    integer part repeats every index, the fractional part draws that share at random.
    :meth:`step` moves the factors between epochs; a caller that never steps keeps the
    mix it started with.

    Args:
        labels: per-sample class label, used to build the per-class index map.
        factor: initial oversampling factor, one value for every class or one per class.
        grow_divisor: a sampled-more class is divided by this, so below 1. Both divisors
            at 1.0 hold the mix wherever ``factor`` put it.
        shrink_divisor: a sampled-less class is divided by this, so above 1.
        overfit_margin: val-train gap, in train-loss sigmas, past which a hard class stops
            growing. 0.0 stops any class whose val loss exceeds its train loss.
        bounds: (low, high) clip applied to any factors :meth:`set_factors` is given.
        num_classes: class count, inferred from ``labels`` when None, which assumes
            every class from 0 to the largest label is present.
        num_samples: epoch length; defaults to the initial oversampled index count.
        generator (Generator): Generator used in sampling.
    """

    indices: Sequence[int]

    def __init__(
        self,
        labels,
        factor: float | Sequence[float] = 1.0,
        *,
        grow_divisor: float = GROW_DIVISOR,
        shrink_divisor: float = SHRINK_DIVISOR,
        overfit_margin: float = OVERFIT_MARGIN,
        bounds: tuple[float, float] = (0.8, 100.0),
        num_classes: int | None = None,
        num_samples: int | None = None,
        generator: torch.Generator | None = None,
    ) -> None:
        self.labels = labels
        self.grow_divisor = grow_divisor
        self.shrink_divisor = shrink_divisor
        self.overfit_margin = overfit_margin
        self.bounds = bounds
        self.generator = generator

        if num_classes is None:
            num_classes = int(torch.max(self.labels) + 1)

        # get inverse map for labels
        self.class_idx = [torch.where(self.labels == i)[0] for i in range(num_classes)]

        factors = torch.as_tensor(factor, dtype=torch.float32)
        if factors.ndim == 0:
            factors = factors.repeat(num_classes)
        if factors.shape != (num_classes,):
            raise ValueError(
                f"factor must be a scalar or have {num_classes} entries, "
                f"got shape {tuple(factors.shape)}"
            )
        # uniform by default, so the epoch is longer but keeps the natural class mix;
        # stepping is what moves the classes apart from there
        self.set_factors(factors)
        # pinned: stepping varies the class mix, not the epoch length, which the LR
        # schedule sizes T_max from once at setup
        self.num_samples = len(self.indices) if num_samples is None else num_samples

    @property
    def factors(self) -> torch.Tensor:
        return self._factors

    def set_factors(self, factors) -> None:
        """Clip to ``bounds`` and rebuild the index list."""
        self._factors = torch.clip(torch.as_tensor(factors, dtype=torch.float32), *self.bounds)
        self.indices = self.oversample(self._factors)

    def step(self, train_loss, train_labels, val_loss, val_labels) -> None:
        """Move every class's factor from one epoch's losses, then rebuild the index list.

        Above-average train loss is sampled more unless the val-train gap exceeds
        ``overfit_margin`` sigmas, below-average less, and a class missing from either
        split keeps its factor.
        """
        # a copy, so a rule that raises midway cannot leave the sampler half-updated
        factors = self._factors.clone()
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

            # not if/else: a degenerate epoch, sigma 0 or nan, has to move nothing
            if undertrained_score > 0.0:
                if overfitting_score < self.overfit_margin * global_std_train_loss:
                    factors[i] = factors[i] / self.grow_divisor
            elif undertrained_score < 0.0:
                factors[i] = factors[i] / self.shrink_divisor

        self.set_factors(factors)

    def oversample(self, factors):
        indices = []
        for i in range(len(self.class_idx)):
            class_idx = self.class_idx[i]
            num_samples = class_idx.size(0)

            class_factor = factors[i]
            class_factor_n, class_factor_f = int(class_factor), class_factor % 1.0

            indices.append(torch.repeat_interleave(class_idx, class_factor_n))
            indices.append(
                class_idx[
                    torch.randperm(num_samples, generator=self.generator)[
                        : int(num_samples * class_factor_f)
                    ]
                ]
            )
        return torch.cat(indices)

    def __iter__(self) -> Iterator[int]:
        order = torch.randperm(len(self.indices), generator=self.generator)[: len(self)]
        return (self.indices[i] for i in order)

    def __len__(self) -> int:
        return min(self.num_samples, len(self.indices))
