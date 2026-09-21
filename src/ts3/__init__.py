"""Task Suite 3: Brain Region Prediction.

10-class Cosmos-level region classification (Allen CCF) from individual unit
activity, spanning coarse but functionally meaningful regions such as Isocortex,
Hippocampus, and Cerebellum. Evaluated in single-unit and multi-unit settings,
zero-shot on held-out animals. Two regimes probe different generalization axes:
transductive zero-shot (adaptation via non-region supervision) and inductive
zero-shot (no adaptation, testing invariance across sessions, probes, and animals).
Performance is measured with macro-averaged F1.

Which regime a model can be evaluated under is a property of the model, not a setting:
see ``ts3.models.inductive`` and ``ts3.models.transductive``.
"""

from core.features import compute_isi_histogram

from .models.base import Extractor
from .ts3_dataset import IBLBrainWideBenchTS3

__all__ = [
    "Extractor",
    "IBLBrainWideBenchTS3",
    "compute_isi_histogram",
]

__api_ref__ = {
    "description": None,
    "sections": [
        {
            "title": "Dataset",
            "autosummary": ["IBLBrainWideBenchTS3"],
        },
        {
            "title": "Extraction",
            "autosummary": ["Extractor"],
        },
        {
            "title": "ISI histograms",
            "autosummary": ["compute_isi_histogram"],
        },
    ],
}
