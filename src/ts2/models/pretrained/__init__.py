"""Eval trainers that predict activity from a pretrained encoder."""

from .mtm import MtMEvalTrainer
from .ndt_stitch import NDTStitchEvalTrainer

__all__ = [
    "MtMEvalTrainer",
    "NDTStitchEvalTrainer",
]

__api_ref__ = {
    "description": None,
    "sections": [
        {
            "title": None,
            "autosummary": ["MtMEvalTrainer", "NDTStitchEvalTrainer"],
        },
    ],
}
