"""Baselines fit on one session, from scratch."""

from .cebra import CEBRA, CEBRAEvalTrainer
from .gru import GRU
from .linear import Linear
from .mlp import MLP
from .ndt_superv import NDTSuperv
from .tcn import TCN

__all__ = [
    "CEBRA",
    "GRU",
    "MLP",
    "TCN",
    "CEBRAEvalTrainer",
    "Linear",
    "NDTSuperv",
]

__api_ref__ = {
    "description": None,
    "sections": [
        {
            "title": None,
            "autosummary": [
                "Linear",
                "MLP",
                "GRU",
                "TCN",
                "CEBRA",
                "CEBRAEvalTrainer",
                "NDTSuperv",
            ],
        },
    ],
}
