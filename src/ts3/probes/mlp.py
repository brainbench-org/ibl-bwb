"""The MLP probe: an Optuna sweep over subject-wise folds, refit on all pretrain units."""

import os
from typing import Any, Literal

import numpy as np
import pandas as pd
import ray
import torch
import torch.nn.functional as F
from ray import tune
from ray.tune.search.optuna.optuna_search import OptunaSearch
from sklearn.metrics import f1_score
from sklearn.preprocessing import StandardScaler
from torch import nn
from tqdm import tqdm

from core.optim import param_groups
from core.utils.logger import get_cli_logger
from core.utils.util import kfold_assignment, seed_everything
from ibl_bwb_eval.tasks import TS3ReadoutSpec, check_ts3_label_order
from ts3.probes.base import Probe

logger = get_cli_logger()

# Deliberately not a config group: the space is part of what the MLP probe is, so two
# submissions are compared on the same search rather than on how wide each one made it.
SEARCH_SPACE = {
    # model
    "m.hidden_layers": tune.randint(1, 4),
    "m.hidden_dim_log2": tune.randint(5, 9),
    "m.dropout": tune.quniform(0, 0.6, 0.2),
    "m.batch_norm": tune.choice([True, False]),
    "m.activation": tune.choice(["relu", "gelu", "tanh"]),
    # training
    "num_epochs": tune.qrandint(50, 200, 50),
    "batch_size_log2": tune.randint(7, 10),
    "lr": tune.loguniform(1e-5, 1e-2),
    "wd": tune.loguniform(1e-5, 1e-1),
}


def _seed_everything(seed: int):
    seed_everything(seed)
    # read at interpreter start, so this is for the ray workers, which are child processes
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.backends.cudnn.deterministic = True


def build_mlp(
    in_features: int,
    n_classes: int,
    hidden_dim: int,
    hidden_layers: int,
    dropout: float,
    activation: Literal["relu", "gelu", "tanh"],
    batch_norm: bool,
) -> nn.Module:
    assert hidden_layers >= 1

    def norm() -> nn.Module:
        if batch_norm:
            return nn.BatchNorm1d(hidden_dim)
        return nn.Identity()

    _ACT_MAP = {"relu": nn.ReLU, "gelu": nn.GELU, "tanh": nn.Tanh}
    act = _ACT_MAP[activation]

    layers = []
    for i in range(hidden_layers):
        in_dim = in_features if i == 0 else hidden_dim
        layer = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            norm(),
            act(),
            nn.Dropout(dropout),
        )
        layers.append(layer)

    layers.append(nn.Linear(hidden_dim, n_classes))
    return nn.Sequential(*layers)


def train_one_mlp(x_train, y_train, hparams, device, n_classes, grad_clip, seed):
    """Fit one MLP, ``seed`` fixing its init, batch order and dropout masks.

    A Ray worker starts with a seed of its own, so this one travels with the work.
    """
    torch.manual_seed(seed)
    train_ds = torch.utils.data.TensorDataset(x_train, y_train)
    train_dl = torch.utils.data.DataLoader(
        train_ds,
        batch_size=2 ** hparams["batch_size_log2"],
        shuffle=True,
        drop_last=True,
    )
    _, counts = y_train.squeeze().unique(sorted=True, return_counts=True)
    assert len(counts) == n_classes, (
        f"fold covers {len(counts)} of {n_classes} classes; the class weights and the macro "
        "F1 both need every class present in the fold"
    )
    class_weights = counts.min() / counts
    class_weights = class_weights.to(device)

    model = build_mlp(
        in_features=x_train.size(1),
        n_classes=n_classes,
        hidden_dim=2 ** hparams["m.hidden_dim_log2"],
        hidden_layers=hparams["m.hidden_layers"],
        dropout=hparams["m.dropout"],
        activation=hparams["m.activation"],
        batch_norm=hparams["m.batch_norm"],
    ).to(device)

    optim = torch.optim.AdamW(
        params=param_groups(model, weight_decay=hparams["wd"]),
        lr=hparams["lr"],
    )

    lr_sched = torch.optim.lr_scheduler.CosineAnnealingLR(
        optim,
        T_max=len(train_dl) * hparams["num_epochs"],
        eta_min=hparams["lr"] * 1e-2,
    )

    pbar = tqdm(range(hparams["num_epochs"]), desc="Train epoch", leave=False)
    for _ in pbar:
        model.train()
        for x, y in train_dl:
            x, y = x.to(device), y.to(device)
            pred = model(x)
            loss = F.cross_entropy(pred, y.squeeze(), weight=class_weights)

            optim.zero_grad()
            loss.backward()
            if grad_clip is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optim.step()
            lr_sched.step()

    return model


def tune_func(hparams: dict[str, Any], X, Y, kfold_idx, n_classes, num_folds, grad_clip, seed):
    dev = torch.device("cuda")

    metric_list = []
    for fold_idx in range(num_folds):
        # On purpose fit on the one fold, score on the other four.
        # the winning hparams are refit on all pretrain units in fit_predict()
        train_mask = kfold_idx == fold_idx

        # Train
        x_train, y_train = X[train_mask].clone(), Y[train_mask].clone()
        # one seed per fold, so two configs are compared on the same draw
        model = train_one_mlp(x_train, y_train, hparams, dev, n_classes, grad_clip, seed + fold_idx)

        # Eval
        x_val, y_val = X[~train_mask].clone(), Y[~train_mask].clone()
        model.eval()
        with torch.inference_mode():
            pred = model(x_val.to(dev)).cpu().numpy()

        pred_label = np.argmax(pred, axis=1)
        f1 = f1_score(y_val, pred_label, average="macro")
        metric_list.append(f1)

    mean_f1 = np.mean(metric_list)
    tune.report({"val_f1": mean_f1})


class MLPProbe(Probe):
    r"""An MLP whose hyperparameters are swept with Optuna before the final fit.

    Args:
        num_trials: sweep budget.
        num_folds: subject-wise folds; each trial fits on one and scores on the rest.
        num_concurrent: trials in flight at once.
        seed: the search, every trial and the final fit. Not the pretraining seed a
            submission is filed under, and not trial completion order, which is unseeded.
        gpu_per_trial: fractional GPU per trial.
        cpu_per_trial: CPUs per trial.
        grad_clip: max grad norm, ``null`` to disable. Fixed across trials, not swept.
    """

    def __init__(
        self,
        num_trials: int = 100,
        num_folds: int = 5,
        num_concurrent: int = 4,
        seed: int = 42,
        gpu_per_trial: float = 0.25,
        cpu_per_trial: float = 1,
        grad_clip: float | None = 1.0,
    ):
        self.num_trials = num_trials
        self.num_folds = num_folds
        self.num_concurrent = num_concurrent
        self.seed = seed
        self.gpu_per_trial = gpu_per_trial
        self.cpu_per_trial = cpu_per_trial
        self.grad_clip = grad_clip

    def fit_predict(
        self,
        train_embs: np.ndarray,
        train_md: pd.DataFrame,
        eval_embs: np.ndarray,
        spec: TS3ReadoutSpec,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        assert torch.cuda.is_available(), (
            "MLPProbe trains on a GPU; probe=linear is the CPU-only probe"
        )
        _seed_everything(self.seed)

        scaler = StandardScaler()
        train_embs = scaler.fit_transform(train_embs)
        eval_embs = scaler.transform(eval_embs)
        check_ts3_label_order(sorted(train_md.brain_region.unique()), spec.id)

        label_to_idx = {label: i for i, label in enumerate(spec.label_names)}
        train_X = torch.tensor(train_embs).to(torch.float32)
        train_Y = torch.tensor([label_to_idx[x] for x in train_md.brain_region.to_numpy()])[:, None]
        kfold_idx = torch.tensor(kfold_assignment(train_md.subject_id.values, self.num_folds))

        ray.init(
            address="local",
            log_to_driver=False,
            ignore_reinit_error=True,
            num_cpus=len(os.sched_getaffinity(0)),
            num_gpus=torch.cuda.device_count(),
        )
        try:
            tuner = tune.Tuner(
                tune.with_resources(
                    tune.with_parameters(
                        tune_func,
                        X=train_X,
                        Y=train_Y,
                        kfold_idx=kfold_idx,
                        n_classes=spec.dim,
                        num_folds=self.num_folds,
                        grad_clip=self.grad_clip,
                        seed=self.seed,
                    ),
                    resources={"gpu": self.gpu_per_trial, "cpu": self.cpu_per_trial},
                ),
                tune_config=tune.TuneConfig(
                    num_samples=self.num_trials,
                    search_alg=OptunaSearch(seed=self.seed),
                    metric="val_f1",
                    mode="max",
                    max_concurrent_trials=self.num_concurrent,
                ),
                param_space=SEARCH_SPACE,
            )
            best = tuner.fit().get_best_result()
        finally:
            ray.shutdown()

        best_hparams = best.config
        fit_metrics = {"val_f1": best.metrics["val_f1"]}
        fit_metrics.update({f"hparams/{k}": v for k, v in best_hparams.items()})
        logger.info(f"Best val_f1: {best.metrics['val_f1']}")
        logger.info(f"Best hparams: {best_hparams}")

        dev = torch.device("cuda")
        model = train_one_mlp(
            train_X, train_Y, best_hparams, dev, spec.dim, self.grad_clip, self.seed
        )
        model.eval()
        with torch.inference_mode():
            logits = model(torch.tensor(eval_embs).to(torch.float32).to(dev))
            return F.softmax(logits, dim=-1).cpu().numpy(), fit_metrics
