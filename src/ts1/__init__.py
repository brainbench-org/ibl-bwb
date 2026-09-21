"""Task Suite 1: Behavior Prediction.

Eight decoding tasks from the IBL Brainwide Map decision-making task: five
timestep-level regression tasks (licking rate, whisker motion energy,
wheel speed, left/right paw speed at 50 Hz) and three sequence-level
classification tasks (stimulus contrast, choice, reward). Evaluation uses a
causal 40/20/40 temporal split on held-out animals, testing robustness to
session nonstationarities and shifting probes.
"""

from ts1.ts1_dataset import IBLBrainWideBenchTS1
from ts1.ts1_eval_trainer import TS1EvalTrainer
from ts1.ts1_test_mixin import TS1TestMixin

__all__ = [
    "IBLBrainWideBenchTS1",
    "TS1EvalTrainer",
    "TS1TestMixin",
]

__api_ref__ = {
    "description": None,
    "sections": [
        {
            "title": None,
            "autosummary": ["IBLBrainWideBenchTS1", "TS1TestMixin", "TS1EvalTrainer"],
        },
    ],
}
