"""The pretrained encoders, their maskers, and the pretrainers that fit them."""

from .mtm import MtM, MtMMasker, MtMPretrain
from .ndt2 import NDT2, NDT2Masker, NDT2Pretrain
from .ndt_stitch import NDTStitch, NDTStitchMasker, NDTStitchPretrain
from .neds import NEDS, NEDSMasker, NEDSPretrain
from .nemo import NEMO, ACGEncoder, LinearProjector, NEMOPretrain, WVFEncoder
from .nuclr import NuCLR, NuCLRLoss, NuCLRPretrain
from .possm import POSSM, POSSMMultitaskPretrain, POSSMSingleTaskPretrain
from .poyo import POYO, POYOSingleTaskPretrain
from .poyo_plus import POYOPlus, POYOPlusMultitaskPretrain
from .rrr import RRRDecoder, RRRSingleTaskPretrain

__all__ = [
    "NDT2",
    "NEDS",
    "NEMO",
    "POSSM",
    "POYO",
    "ACGEncoder",
    "LinearProjector",
    "MtM",
    "MtMMasker",
    "MtMPretrain",
    "NDT2Masker",
    "NDT2Pretrain",
    "NDTStitch",
    "NDTStitchMasker",
    "NDTStitchPretrain",
    "NEDSMasker",
    "NEDSPretrain",
    "NEMOPretrain",
    "NuCLR",
    "NuCLRLoss",
    "NuCLRPretrain",
    "POSSMMultitaskPretrain",
    "POSSMSingleTaskPretrain",
    "POYOPlus",
    "POYOPlusMultitaskPretrain",
    "POYOSingleTaskPretrain",
    "RRRDecoder",
    "RRRSingleTaskPretrain",
    "WVFEncoder",
]


__api_ref__ = {
    "description": (
        "Each model is one directory under ``src/pretrain/models/``, pairing an encoder with "
        "the pretrainer that fits it, and a masker where the objective needs one. This page "
        "is grouped the same way, one section per directory."
    ),
    "sections": [
        {
            "title": "NDT Stitch",
            "autosummary": ["NDTStitch", "NDTStitchMasker", "NDTStitchPretrain"],
        },
        {
            "title": "MtM",
            "autosummary": ["MtM", "MtMMasker", "MtMPretrain"],
        },
        {
            "title": "NDT2",
            "autosummary": ["NDT2", "NDT2Masker", "NDT2Pretrain"],
        },
        {
            "title": "NEDS",
            "autosummary": ["NEDS", "NEDSMasker", "NEDSPretrain"],
        },
        {
            "title": "POYO",
            "autosummary": ["POYO", "POYOSingleTaskPretrain"],
        },
        {
            "title": "POYO+",
            "autosummary": ["POYOPlus", "POYOPlusMultitaskPretrain"],
        },
        {
            "title": "POSSM",
            "description": "Pretrained on one behavior or on several at once, hence two trainers.",
            "autosummary": ["POSSM", "POSSMSingleTaskPretrain", "POSSMMultitaskPretrain"],
        },
        {
            "title": "RRR",
            "autosummary": ["RRRDecoder", "RRRSingleTaskPretrain"],
        },
        {
            "title": "NuCLR",
            "description": "A unit encoder for TS3, trained against the contrastive loss listed here.",
            "autosummary": ["NuCLR", "NuCLRLoss", "NuCLRPretrain"],
        },
        {
            "title": "NEMO",
            "description": "A unit encoder for TS3, assembled from the submodules listed here.",
            "autosummary": [
                "NEMO",
                "NEMOPretrain",
                "WVFEncoder",
                "ACGEncoder",
                "LinearProjector",
            ],
        },
    ],
}
