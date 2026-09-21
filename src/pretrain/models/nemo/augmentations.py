"""NEMO augmentation recipes for waveforms and ACGs.

Ported from NEMO (Yu et al., ICLR 2025):
  - WVF: GaussianNoise (p=0.3), AmpJitter (p=0.4) [electrode_dropout excluded for 1-ch]
  - ACG: temporal_gaussian_smoothing (p=0.5), temporal_jittering (p=0.5),
         amplitude_scaling (p=0.5), additive_gaussian_noise (p=0.5),
         additive_pepper_noise (p=0.5)

The primitives themselves are model-agnostic and live in :mod:`core.transforms`.
"""

from torch_brain.transforms import Compose

from core.transforms.augmentations import (
    AdditiveGaussianNoise,
    AdditivePepperNoise,
    AmplitudeScaling,
    GaussianNoise,
    RandomApply,
    TemporalGaussianSmoothing,
    TemporalJittering,
)


def get_wf_transform(
    aug_p_dict: dict | None = None,
    std: float = 0.1,
) -> Compose:
    """Return waveform augmentation pipeline (GaussianNoise + amplitude jitter)."""
    if aug_p_dict is None:
        aug_p_dict = {"gaussian_noise": 0.3, "amp_jitter": 0.4}
    return Compose(
        [
            RandomApply(GaussianNoise(std=std), p=aug_p_dict["gaussian_noise"]),
            RandomApply(AmplitudeScaling(), p=aug_p_dict["amp_jitter"]),
        ]
    )


def get_acg_transform(
    aug_p_dict: dict | None = None,
) -> Compose:
    """Return ACG augmentation pipeline."""
    if aug_p_dict is None:
        aug_p_dict = {
            "temporal_gaussian_smoothing": 0.5,
            "temporal_jittering": 0.5,
            "amplitude_scaling": 0.5,
            "additive_gaussian_noise": 0.5,
            "additive_pepper_noise": 0.5,
        }
    return Compose(
        [
            RandomApply(TemporalGaussianSmoothing(), p=aug_p_dict["temporal_gaussian_smoothing"]),
            RandomApply(TemporalJittering(), p=aug_p_dict["temporal_jittering"]),
            RandomApply(AmplitudeScaling(), p=aug_p_dict["amplitude_scaling"]),
            RandomApply(AdditiveGaussianNoise(), p=aug_p_dict["additive_gaussian_noise"]),
            RandomApply(AdditivePepperNoise(), p=aug_p_dict["additive_pepper_noise"]),
        ]
    )
