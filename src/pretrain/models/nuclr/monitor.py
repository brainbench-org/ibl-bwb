"""NuCLR's online probe: how well its unit embeddings separate brain regions.

A training diagnostic and not a benchmark number. Every val epoch it fits a logistic
probe over pretrain units, holding out whole subject groups at a time, and logs the
result; nothing selects on it. TS3 scores saved embeddings under its own rules, so this
is a deliberate copy rather than shared code: the table below reads the pretrain regime
and has no eval branch to reach for, which is what keeps the scored units out of a
pretraining run.
"""

import os
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import ray
import torch
import wandb
from matplotlib import pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from core.dataset import IBLBrainWideBench2026, UnitQCPolicy
from core.nn.metrics import rankme
from core.utils.logger import get_cli_logger
from core.utils.util import kfold_assignment
from ibl_bwb_eval.tasks import TS3Task, get_ts3_readout_spec

logger = get_cli_logger()

# Subjects are pooled into this many folds, so one fold holds out several animals at once.
SUBJECT_FOLDS = 5

# The build applies all but the alignment cut; re-stating them keeps the table honest
# if a later build stops. A region label is only as good as its histology.
_PRETRAIN_QC = UnitQCPolicy(
    keep_qc_neural=("PASS", "WARNING"),
    keep_qc_neural_alignment=("PASS",),
    min_firing_rate=1.0,
    min_label=1.0,
)

# Pretrain units left after the cuts above, as a guard on a silent data change.
_EXPECTED_UNITS: dict[str, int] = {"unit_cosmos": 41_834}


@ray.remote(num_cpus=1)
def _fit_fold(X: np.ndarray, y: np.ndarray, X_test: np.ndarray, y_test: np.ndarray) -> dict:
    """One class-balanced logistic fit, scored on the fold it held out."""
    model = Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "classifier",
                LogisticRegression(max_iter=1_000, solver="newton-cg", class_weight="balanced"),
            ),
        ]
    )
    model.fit(X, y)
    return {"preds": model.predict(X_test), "targets": y_test}


def _pretrain_unit_table(data_root: Path, task: TS3Task) -> pd.DataFrame:
    """Region and subject of every pretrain unit worth probing.

    Only the two columns the probe reads, and only the pretrain regime: a run that
    cannot name the eval units cannot be tuned on them.
    """
    spec = get_ts3_readout_spec(task)
    ds = IBLBrainWideBench2026(
        root=str(data_root),
        split=None,  # a unit table is not a temporal sample
        regime="pretrain",
        contract="online brain-region probe",
    )

    frames = []
    for rid in ds.recording_ids:
        u = ds.get_recording(rid).units
        # is the unit's data usable, and is there a target in this task's vocabulary
        keep = _PRETRAIN_QC.keep_mask(u)
        keep &= np.isin(getattr(u, spec.region_key).astype(str), spec.label_names)

        frames.append(
            pd.DataFrame(
                {
                    "unit_id": u.id.astype(str)[keep],
                    "brain_region": getattr(u, spec.region_key).astype(str)[keep],
                    "subject_id": ds.get_recording(rid).subject.id,
                }
            )
        )
    md = pd.concat(frames).set_index("unit_id")

    expected = _EXPECTED_UNITS[task]
    if len(md) != expected:
        raise RuntimeError(f"expected {expected} {task} pretrain units, got {len(md)}")
    return md


class BrainRegionMonitor:
    """Leave-subject-group-out region probe over NuCLR's pretrain-unit embeddings.

    Args:
        data_root: root of the dataset build the run trains on.
        world_size: sets the parallel fold workers, ``world_size * 7``.
        task: which region vocabulary to score. Defaulted, since this is a monitor
            rather than a submission path.
    """

    def __init__(self, data_root: Path, world_size: int, task: TS3Task = "unit_cosmos"):
        self.prefix = "train/l1subout_br"
        self.labels = np.asarray(get_ts3_readout_spec(task).label_names)
        self.train_md = _pretrain_unit_table(Path(data_root), task)
        logger.info(f"train_md size: {len(self.train_md)}")

        os.environ["RAY_ACCEL_ENV_VAR_OVERRIDE_ON_ZERO"] = "0"
        ray.init(address="local", num_cpus=world_size * 7, ignore_reinit_error=True)

    def __call__(self, train_embs: torch.Tensor, train_uids: Sequence[str]) -> dict[str, Any]:
        metrics = {"train/rankme": rankme(train_embs)}  # on device, before the host copy

        embs = train_embs.detach().cpu().numpy()
        uids = np.asarray(train_uids).astype(str)

        finite = np.all(np.isfinite(embs), axis=-1)
        if not finite.all():
            logger.info(f"{(~finite).sum()} train_embs are not finite")
        embs, uids = embs[finite], uids[finite]

        known = pd.Index(uids).isin(self.train_md.index)  # faster than np.isin
        embs, uids = embs[known], uids[known]
        md = self.train_md.loc[uids]

        y = md.brain_region.values
        folds = np.asarray(kfold_assignment(md.subject_id.values, SUBJECT_FOLDS))
        futures = [
            _fit_fold.remote(embs[folds != f], y[folds != f], embs[folds == f], y[folds == f])
            for f in np.unique(folds)
        ]
        return metrics | self._score(ray.get(futures))

    def _score(self, folds: list[dict]) -> dict[str, Any]:
        """Pool the folds' held-out predictions and score them as one set."""
        preds = np.concatenate([f["preds"] for f in folds])
        targets = np.concatenate([f["targets"] for f in folds])
        per_class = f1_score(y_true=targets, y_pred=preds, labels=self.labels, average=None)
        return {
            f"{self.prefix}/acc": accuracy_score(targets, preds),
            f"{self.prefix}/bacc": balanced_accuracy_score(targets, preds),
            f"{self.prefix}/f1_macro": f1_score(y_true=targets, y_pred=preds, average="macro"),
            f"{self.prefix}/f1_micro": f1_score(y_true=targets, y_pred=preds, average="micro"),
            f"{self.prefix}/cm": self._confusion_matrix(targets, preds),
            **{
                f"{self.prefix}/f1/{name}": float(v)
                for name, v in zip(self.labels, per_class, strict=True)
            },
        }

    def _confusion_matrix(self, targets: np.ndarray, preds: np.ndarray) -> wandb.Image:
        cm = confusion_matrix(y_true=targets, y_pred=preds, labels=self.labels, normalize="true")
        size = 0.35 * len(self.labels) + 1.5
        fig = plt.figure(figsize=(size, size))
        disp = ConfusionMatrixDisplay(cm, display_labels=self.labels)
        disp.plot(
            cmap="Blues",
            ax=fig.add_subplot(111),
            xticks_rotation=45,
            colorbar=False,
            im_kw={"vmin": 0, "vmax": 1},
            text_kw={"fontsize": 8},
            values_format=".2f",
        )
        disp.figure_.colorbar(disp.im_, ax=disp.ax_, fraction=0.046, pad=0.04)
        disp.figure_.tight_layout()
        image = wandb.Image(disp.figure_)
        plt.close(fig)  # the figure is the run's, not the current one
        return image
