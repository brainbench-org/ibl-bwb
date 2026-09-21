"""Extractors whose unit embedding is a free parameter indexed by unit identity.

A held-out unit has no row until a run creates one, so the eval part of the embeddings file
comes from per-session calibration checkpoints rather than from the pretrain checkpoint. See
``ts3.models.transductive.base`` for what that costs, and ``ts3.models.inductive`` for the
models it does not apply to. Each model's calibration trainer, which produces those
checkpoints, lives alongside its extractor and is exported here too, even though it is driven
by ``ts3/train.py`` via its Hydra config (``trainer=<name>_calibrate``) rather than
instantiated directly.
"""

from .base import TransductiveExtractor
from .mtm import MtMCalibrateTrainer, MtMExtractor
from .ndt_stitch import NDTStitchCalibrateTrainer, NDTStitchExtractor
from .possm import POSSMCalibrateTrainer, POSSMExtractor
from .poyo_plus import POYOPlusCalibrateTrainer, POYOPlusExtractor

__all__ = [
    "MtMCalibrateTrainer",
    "MtMExtractor",
    "NDTStitchCalibrateTrainer",
    "NDTStitchExtractor",
    "POSSMCalibrateTrainer",
    "POSSMExtractor",
    "POYOPlusCalibrateTrainer",
    "POYOPlusExtractor",
    "TransductiveExtractor",
]

__api_ref__ = {
    "description": None,
    "sections": [
        {
            "title": None,
            "autosummary": [
                "TransductiveExtractor",
                "POYOPlusExtractor",
                "POSSMExtractor",
                "NDTStitchExtractor",
                "MtMExtractor",
            ],
        },
        {
            "title": "Calibration trainers",
            "autosummary": [
                "NDTStitchCalibrateTrainer",
                "MtMCalibrateTrainer",
                "POSSMCalibrateTrainer",
                "POYOPlusCalibrateTrainer",
            ],
        },
    ],
}
