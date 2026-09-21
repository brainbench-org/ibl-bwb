"""Per-unit mean rate: each neuron's own train-window average, constant in time.

Nothing to fit beyond that average and nothing to select, so this is the floor
the other methods are measured against. It is also what they degenerate to on the
task they do not apply to, which is why the mechanism lives in ``StatBaseline``
and this class only names it.
"""

import logging

from .base import StatBaseline

log = logging.getLogger(__name__)


class MeanRate(StatBaseline):
    """Per-unit mean rate, constant in time. Applies to both tasks."""

    TASK = None  # applies everywhere; never degenerates
    NAME = "mean_rate"

    def fit(self, train_dataset, val_dataset):
        log.info(f"{self.NAME}: per-unit mean rate, nothing to fit or select")
