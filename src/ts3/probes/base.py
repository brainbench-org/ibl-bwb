"""What a TS3 probe is: fit on the pretrain units, score the eval units.

Everything downstream of :meth:`Probe.fit_predict` (the multi-unit readout, the reports,
the submission files, W&B) belongs to ``eval.py``, so a probe is only ever responsible
for turning embeddings into probabilities.
"""

from abc import ABC, abstractmethod
from typing import Any

import numpy as np
import pandas as pd

from ibl_bwb_eval.tasks import TS3ReadoutSpec


class Probe(ABC):
    """A classifier fit on unit embeddings, scored by ``eval.py``."""

    @abstractmethod
    def fit_predict(
        self,
        train_embs: np.ndarray,
        train_md: pd.DataFrame,
        eval_embs: np.ndarray,
        spec: TS3ReadoutSpec,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        """Fit on the pretrain units, return eval probabilities and what the fit learned.

        The probabilities have shape (units, classes); the dict is whatever the fit is
        worth reporting about itself, empty for a probe with nothing to say.

        Probabilities, not logits, and that is load-bearing: the multi-unit readout means
        neighbouring units together (:func:`ibl_bwb_eval.multi_unit.multi_unit_prediction`),
        and that rule is defined on probabilities. Columns follow ``spec.label_names``;
        ``train_md`` is the TS3 unit table, so a probe that needs more than the region
        label (subject id, for a grouped split) reads it from there.
        """
