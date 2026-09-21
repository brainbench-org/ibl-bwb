"""Unit selection and spike-count augmentations, shared by every task suite."""

from .augmentations import (
    AdditiveGaussianNoise,
    AdditivePepperNoise,
    AmplitudeScaling,
    GaussianNoise,
    RandomApply,
    TemporalGaussianSmoothing,
    TemporalJittering,
)
from .unit_mask import apply_unit_mask_
from .unit_transforms import FilterUnits, UnitDropout

__all__ = [
    "AdditiveGaussianNoise",
    "AdditivePepperNoise",
    "AmplitudeScaling",
    "FilterUnits",
    "GaussianNoise",
    "RandomApply",
    "TemporalGaussianSmoothing",
    "TemporalJittering",
    "UnitDropout",
    "apply_unit_mask_",
]

__api_ref__ = {
    "description": None,
    "sections": [
        {
            "title": "Unit transforms",
            "autosummary": ["FilterUnits", "UnitDropout", "apply_unit_mask_"],
        },
        {
            "title": "Augmentations",
            "autosummary": [
                "AdditiveGaussianNoise",
                "AdditivePepperNoise",
                "AmplitudeScaling",
                "GaussianNoise",
                "RandomApply",
                "TemporalGaussianSmoothing",
                "TemporalJittering",
            ],
        },
    ],
}
