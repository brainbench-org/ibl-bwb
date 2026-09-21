"""Models trained in-suite on the region labels, which emit probabilities, not embeddings.

They skip ``ts3/extract.py`` and the probes entirely and write their own submission, so the
inductive/transductive question does not arise: what sets them apart is supervision on the
scored label, not adaptation on the eval sessions.
"""

from .lolcat import (
    LOLCAT,
    LOLCATTrainer,
    LossFeedbackSampler,
    MultiHeadGlobalAttention,
)

__all__ = ["LOLCAT", "LOLCATTrainer", "LossFeedbackSampler", "MultiHeadGlobalAttention"]

__api_ref__ = {
    "description": None,
    "sections": [
        {"title": None, "autosummary": ["LOLCAT", "MultiHeadGlobalAttention"]},
        {"title": "Trainers", "autosummary": ["LOLCATTrainer", "LossFeedbackSampler"]},
    ],
}
