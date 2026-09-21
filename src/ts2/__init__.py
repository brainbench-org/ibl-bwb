"""Task Suite 2: Neural Activity Prediction.

Co-smoothing probes spatial structure by reconstructing randomly masked neurons
from the observed population; forecasting probes temporal dynamics by predicting
the last 200 ms of each window from past activity. Both tasks are evaluated with
Poisson D² and bits-per-spike. Splits use interleaved 5-minute blocks (40/20/40)
rather than a strict causal split, reducing nonstationarity effects on neuron
identity across long timescales.
"""

from ts2.ts2_dataset import IBLBrainWideBenchTS2
from ts2.ts2_eval_trainer import TS2EvalTrainer
from ts2.ts2_test_mixin import TS2TestMixin

__all__ = ["IBLBrainWideBenchTS2", "TS2EvalTrainer", "TS2TestMixin"]

__api_ref__ = {
    "description": None,
    "sections": [
        {
            "title": None,
            "autosummary": ["IBLBrainWideBenchTS2", "TS2TestMixin", "TS2EvalTrainer"],
        },
    ],
}
