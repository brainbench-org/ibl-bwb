"""What a pretraining run reads, shared by the models that fit the same objective."""

from .mask_modeling_spike import IBLBrainWideBenchMaskModelingSpikes
from .multi_task_behavior import IBLBrainWideBenchMultiTaskBehavior
from .single_task_behavior import IBLBrainWideBenchSingleTaskBehavior

__all__ = [
    "IBLBrainWideBenchMaskModelingSpikes",
    "IBLBrainWideBenchMultiTaskBehavior",
    "IBLBrainWideBenchSingleTaskBehavior",
]


__api_ref__ = {
    "description": None,
    "sections": [
        {
            "title": None,
            "autosummary": [
                "IBLBrainWideBenchMaskModelingSpikes",
                "IBLBrainWideBenchSingleTaskBehavior",
                "IBLBrainWideBenchMultiTaskBehavior",
            ],
        },
    ],
}
