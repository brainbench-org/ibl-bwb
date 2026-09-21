"""Stage 3: one unit-embeddings file in, scores and a submission out.

Fitting the probe on the pretrain units and their labels is the protocol; the eval labels are
only ever scored against. Everything a probe does not own lives here rather than inside it:
the two scopes it is scored at, the multi-unit readout, the reports and the submission files,
so two probes are compared on one treatment of their outputs. Which probe runs is a config
choice (``probe=``), resolved the way ``extractor=`` is in ``ts3/extract.py``.
"""

import hydra
import numpy as np
import torch
import wandb
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig
from sklearn.metrics import classification_report

from core.launch import run
from core.utils.logger import Logger
from core.utils.util import expand_path
from ibl_bwb_eval.entity_ids import encode_entity_ids
from ibl_bwb_eval.multi_unit import multi_unit_prediction
from ibl_bwb_eval.predictions import PredictionsWriter
from ibl_bwb_eval.tasks import get_ts3_readout_spec, task_id
from ts3.embeddings import load_embs_and_md


def _submission_seed(cfg: DictConfig, file_seed: int | None) -> int:
    """The seed a submission is filed under.

    The embeddings file's own when it recorded one, otherwise ``cfg.seed``.
    """
    if file_seed is None:
        assert cfg.seed is not None, (
            "seed is required to write a submission: this embeddings file records no run "
            "seed, so pass the seed of the run that produced it as seed="
        )
        return int(cfg.seed)
    if cfg.seed is not None and int(cfg.seed) != int(file_seed):
        raise ValueError(
            f"seed={cfg.seed} disagrees with the embeddings file, which run seed "
            f"{file_seed} produced. Drop seed= to file the submission under the file's own."
        )
    return int(file_seed)


@hydra.main(version_base="1.2", config_path="./configs", config_name="eval.yaml")
def main(cfg: DictConfig):
    assert cfg.data_root, (
        "data_root is unset: set BWB_DATA_ROOT, its per-build override, or pass data_root="
    )
    if cfg.save_preds.enable:
        assert cfg.save_preds.label, "save_preds.label is required to write a submission"

    logger = Logger(rank=0, disable_pbar=True)
    logger.init_wandb(cfg.wandb)
    logger.save_config(cfg)

    probe_name = HydraConfig.get().runtime.choices.probe
    spec = get_ts3_readout_spec(cfg.task)
    train_embs, train_md, eval_embs, eval_md = load_embs_and_md(
        expand_path(cfg.data_root), expand_path(cfg.emb_path), cfg.task
    )
    logger.info(f"Train units: {len(train_md)}, Eval units: {len(eval_md)}")
    submission_seed = (
        _submission_seed(cfg, eval_md.attrs["run_seed"]) if cfg.save_preds.enable else None
    )

    probe = hydra.utils.instantiate(cfg.probe)
    pred_proba, fit_metrics = probe.fit_predict(train_embs, train_md, eval_embs, spec)

    # Pooling is defined on probabilities, which is why fit_predict returns them.
    scopes = {
        "single": pred_proba,
        "multi": multi_unit_prediction(
            pred_proba=pred_proba,
            depths=eval_md.depths.values,
            probe_ids=eval_md.probe_ids.values,
        ),
    }

    targets = eval_md.brain_region.values
    names = np.array(spec.label_names)

    result_strs = []
    for scope, proba in scopes.items():
        preds = names[proba.argmax(axis=1)]
        result_strs.append(f"{'-' * 20} {scope.capitalize()} Unit {'-' * 20}")
        result_strs.append(classification_report(targets, preds, digits=5))
        logger.log_dict(
            {
                f"probe/{cfg.task}/{probe_name}_{scope}/{name}": value
                for name, value in spec.score(targets, preds).items()
            }
        )
    logger.log_dict(
        {f"probe/{cfg.task}/{probe_name}/{name}": value for name, value in fit_metrics.items()}
    )

    result_str = "\n".join(result_strs)
    logger.info("\n" + result_str)

    if cfg.save_preds.enable:
        entity_ids = encode_entity_ids(eval_md.index.values)
        for scope, proba in scopes.items():
            writer = PredictionsWriter(
                enable=True,
                base_path=expand_path(cfg.save_preds.path),
                task=task_id("ts3", cfg.task),
                seed=submission_seed,
                label=f"{cfg.save_preds.label}_{probe_name}_{scope}",
                metadata={
                    "label_names": ",".join(spec.label_names),
                    "unit_filtering": eval_md.attrs["unit_filtering"],
                    "dataset_version": eval_md.attrs["dataset_version"],
                },
            )
            writer.set(entity_ids=entity_ids, pred_proba=torch.tensor(proba, dtype=torch.float32))
            writer.save(logger=logger)

    logger.push()
    wandb.finish()


if __name__ == "__main__":
    run(main)
