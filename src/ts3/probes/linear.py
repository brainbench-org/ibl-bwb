"""The logistic-regression probe."""

from typing import Any

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from ibl_bwb_eval.tasks import TS3ReadoutSpec, check_ts3_label_order
from ts3.probes.base import Probe


class LinearProbe(Probe):
    r"""Multinomial logistic regression on standardized embeddings.

    Args:
        balanced: weight classes by inverse frequency.
        C: inverse regularization strength.
        max_iter: solver iteration cap.
    """

    def __init__(self, balanced: bool = True, C: float = 1.0, max_iter: int = 1_000):
        self.balanced = balanced
        self.C = C
        self.max_iter = max_iter

    def fit_predict(
        self,
        train_embs: np.ndarray,
        train_md: pd.DataFrame,
        eval_embs: np.ndarray,
        spec: TS3ReadoutSpec,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        model = Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "classifier",
                    LogisticRegression(
                        C=self.C,
                        max_iter=self.max_iter,
                        solver="newton-cg",
                        class_weight="balanced" if self.balanced else None,
                    ),
                ),
            ]
        )
        model.fit(train_embs, train_md.brain_region.values)
        check_ts3_label_order(model.named_steps["classifier"].classes_, spec.id)
        return model.predict_proba(eval_embs), {}
