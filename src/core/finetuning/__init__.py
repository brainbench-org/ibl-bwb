"""How much of a pretrained encoder a downstream run is allowed to move."""

from core.finetuning.base import FinetuningStrategy
from core.finetuning.full_finetuning import FullFinetuning
from core.finetuning.gradual_unfreezing import GradualUnfreezing
from core.finetuning.probe import Probe

__all__ = ["FinetuningStrategy", "FullFinetuning", "GradualUnfreezing", "Probe"]

__api_ref__ = {
    "description": None,
    "sections": [{"title": None, "autosummary": __all__}],
}
