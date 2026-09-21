"""Interspike-interval features.

Model-agnostic and stateless: the TS3 ISI baseline embeds units with it, and LOLCAT
uses it as its input feature.
"""

import numpy as np


def compute_isi_histogram(
    spike_times: np.ndarray,
    n_bins: int,
    t_min: float,
    t_max: float,
    log_bins: bool = True,
) -> np.ndarray:
    """ISI histogram, L1-normalized to sum=1."""
    if len(spike_times) < 2:
        return np.zeros(n_bins, dtype=np.float32)
    isis = np.diff(spike_times)
    if log_bins:
        bins = np.geomspace(t_min, t_max, n_bins + 1)
    else:
        bins = np.linspace(t_min, t_max, n_bins + 1)
    hist, _ = np.histogram(isis, bins=bins)
    total = hist.sum()
    return (hist / total).astype(np.float32) if total > 0 else hist.astype(np.float32)
