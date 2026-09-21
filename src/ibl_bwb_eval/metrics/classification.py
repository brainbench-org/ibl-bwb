"""The classification metrics a TS3 submission is scored on.

One function per metric, each returning the names it owns: a score per class and the macro
average over classes. The scorer and the probes both call these, so a run's live scores and
its offline scores cannot end up computed differently. A TS3 task that is not classification
declares its own functions on its spec instead of reusing these.
"""

from collections.abc import Callable, Sequence

import numpy as np
from sklearn.metrics import f1_score, precision_score, recall_score

TS3Metric = Callable[[np.ndarray, np.ndarray, Sequence[str]], dict[str, float]]


def _per_class_and_macro(score: Callable, name: str) -> TS3Metric:
    def metric(y_true, y_pred, label_names: Sequence[str]) -> dict[str, float]:
        # labels= pins the vocabulary to the task's, so a class missing from both sides
        # scores 0 rather than dropping out of the result and raising downstream
        labels = list(label_names)
        kwargs = {"labels": labels, "zero_division": 0}
        per_class = score(y_true, y_pred, average=None, **kwargs)
        out = {f"{c}/{name}": float(v) for c, v in zip(labels, per_class, strict=True)}
        out[f"macro/{name}"] = float(score(y_true, y_pred, average="macro", **kwargs))
        return out

    metric.__name__ = f"{name.replace('-', '_')}_metric"
    return metric


precision = _per_class_and_macro(precision_score, "precision")
recall = _per_class_and_macro(recall_score, "recall")
f1 = _per_class_and_macro(f1_score, "f1-score")

# Keyed by the names ``ibl_bwb_eval.tasks.ts3.CLASSIFICATION_METRICS`` declares. That module
# is stdlib-only and cannot import these, so the two are bound by a test instead.
CLASSIFICATION_SCORERS: dict[str, TS3Metric] = {
    "precision": precision,
    "recall": recall,
    "f1-score": f1,
}
