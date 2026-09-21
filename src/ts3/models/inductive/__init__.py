"""Extractors whose unit embedding is a function of the unit's own data.

Nothing about unit identity is stored in the weights, so one frozen checkpoint (or, for ISI,
no checkpoint at all) represents held-out units directly and both halves of the embeddings
file come from the same encoder. See ``ts3.models.transductive`` for the models this is not
true of.
"""

from .isi import ISIExtractor
from .nemo import NEMOExtractor
from .nuclr import NuCLRExtractor

__all__ = ["ISIExtractor", "NEMOExtractor", "NuCLRExtractor"]

__api_ref__ = {
    "description": None,
    "sections": [
        {"title": None, "autosummary": ["NuCLRExtractor", "NEMOExtractor", "ISIExtractor"]},
    ],
}
