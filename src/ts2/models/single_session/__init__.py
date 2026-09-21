"""Baselines fit on one session, from statistical to sequence models."""

from .autoencoder import AutoencoderMLP
from .lfads import LFADS, LFADSEvalTrainer
from .ndt import NDT, NDTEvalTrainer
from .stat_baseline import (
    MeanRate,
    PopCoupling,
    RidgeAR,
    RRRReadout,
    RRRReadoutWithISI,
    Shrinkage,
    StatBaseline,
    StatBaselineTrainer,
    TrailingMean,
)

__all__ = [
    "LFADS",
    "NDT",
    "AutoencoderMLP",
    "LFADSEvalTrainer",
    "MeanRate",
    "NDTEvalTrainer",
    "PopCoupling",
    "RRRReadout",
    "RRRReadoutWithISI",
    "RidgeAR",
    "Shrinkage",
    "StatBaseline",
    "StatBaselineTrainer",
    "TrailingMean",
]

__api_ref__ = {
    "description": None,
    "sections": [
        {
            "title": None,
            "autosummary": [
                "AutoencoderMLP",
                "LFADS",
                "LFADSEvalTrainer",
                "NDT",
                "NDTEvalTrainer",
            ],
        },
        {
            "title": "Statistical baselines",
            "autosummary": [
                "StatBaseline",
                "PopCoupling",
                "RRRReadout",
                "RRRReadoutWithISI",
                "TrailingMean",
                "Shrinkage",
                "RidgeAR",
                "StatBaselineTrainer",
            ],
        },
    ],
}
