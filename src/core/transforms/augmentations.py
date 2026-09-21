"""Array augmentation primitives, shared by unit-feature models.

Covers waveforms, ACGs and ISI histograms. The per-model recipes that compose them live
with their model.
"""

import numpy as np
from scipy.ndimage import gaussian_filter


class RandomApply:
    """Apply a transform with probability p."""

    def __init__(self, transform, p: float = 0.5):
        self.transform = transform
        self.p = p

    def __call__(self, x):
        if np.random.random() < self.p:
            return self.transform(x)
        return x


class AmplitudeScaling:
    """Rescales amplitude by a small random factor."""

    def __init__(self, lo: float = 0.9, hi: float = 1.1):
        self.lo = lo
        self.hi = hi

    def __call__(self, x: np.ndarray) -> np.ndarray:
        return x * np.random.uniform(self.lo, self.hi)


class GaussianNoise:
    """Adds Gaussian noise scaled to the input std."""

    def __init__(self, std: float = 0.1):
        self.std = std

    def __call__(self, x: np.ndarray) -> np.ndarray:
        noise = np.random.normal(0, self.std * np.std(x), size=x.shape)
        return x + noise


class TemporalGaussianSmoothing:
    """Gaussian smoothing along the temporal (last) axis only."""

    def __init__(self, sigma: int = 2):
        self.sigma = sigma

    def __call__(self, x: np.ndarray) -> np.ndarray:
        return gaussian_filter(x, sigma=(0, self.sigma))


class TemporalJittering:
    """Random row-wise circular shift along the temporal axis."""

    def __init__(self, max_jitter: int = 3):
        self.max_jitter = max_jitter

    def __call__(self, x: np.ndarray) -> np.ndarray:
        num_rows, num_cols = x.shape
        jitters = np.random.randint(-self.max_jitter, self.max_jitter + 1, size=num_rows)
        col_indices = (np.arange(num_cols) - jitters[:, np.newaxis]) % num_cols
        return x[np.arange(num_rows)[:, np.newaxis], col_indices]


class AdditiveGaussianNoise:
    """Adds Gaussian noise scaled to the input max, clipped at 0."""

    def __init__(self, mean: float = 0.0, std: float = 0.1):
        self.mean = mean
        self.std = std

    def __call__(self, x: np.ndarray) -> np.ndarray:
        noise = np.random.normal(self.mean, self.std * np.max(x), size=x.shape)
        return np.clip(x + noise, 0, np.inf)


class AdditivePepperNoise:
    """Randomly zeros out bins."""

    def __init__(self, pepper_prob: float = 0.05):
        self.pepper_prob = pepper_prob

    def __call__(self, x: np.ndarray) -> np.ndarray:
        noisy = np.copy(x)
        noisy[np.random.random(x.shape) < self.pepper_prob] = 0.0
        return noisy
