"""Layers shared by more than one model, and the attention backends they run on."""

from .embedding import Embedding
from .init import tfixup_init_
from .multitask_readout import MultitaskReadout
from .varlen_attention import (
    ATTN_IMPLS,
    AttnImpl,
    cross_attn,
    self_attn,
    uses_xformers,
    validate_attn_impl,
)

__all__ = [
    "ATTN_IMPLS",
    "AttnImpl",
    "Embedding",
    "MultitaskReadout",
    "cross_attn",
    "self_attn",
    "tfixup_init_",
    "uses_xformers",
    "validate_attn_impl",
]

__api_ref__ = {
    "description": None,
    "sections": [
        {
            "title": "Layers",
            "autosummary": ["Embedding", "MultitaskReadout"],
        },
        {
            "title": "Initialization",
            "autosummary": ["tfixup_init_"],
        },
        {
            "title": "Variable-length attention",
            "description": (
                "Two interchangeable backends behind one call signature, so a model built "
                "on chained tokens can drop xformers entirely."
            ),
            "autosummary": ["self_attn", "cross_attn", "validate_attn_impl", "uses_xformers"],
        },
    ],
}
