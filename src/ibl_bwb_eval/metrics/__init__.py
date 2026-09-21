"""Metric definitions the benchmark reports.

These are the metrics a submission is scored on, so they live with the scorer rather
than with the training code, and training-time logging imports them from here too: a
change to one moves every number already reported, which importing them by way of
``core`` would hide. ``core.nn.metrics`` keeps only diagnostics that are scored by
nothing.
"""

import torch

from .bps import BPS
from .classification import CLASSIFICATION_SCORERS
from .poisson_d2 import PoissonD2Score

__all__ = ["BPS", "CLASSIFICATION_SCORERS", "PoissonD2Score", "aggregate_metrics"]

__api_ref__ = {
    "description": None,
    "sections": [{"title": None, "autosummary": ["BPS", "PoissonD2Score", "aggregate_metrics"]}],
}


def aggregate_metrics(metrics: dict) -> dict:
    """Compute and flatten a dict of torchmetrics metrics to plain Python scalars."""
    computed = {name: metric.compute() for name, metric in metrics.items()}
    return {
        name: (
            v.mean().item()
            if torch.is_tensor(v) and v.ndim > 0
            else v.item()
            if torch.is_tensor(v)
            else v
        )
        for name, v in computed.items()
    }
