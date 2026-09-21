# src/ts1/scripts/finetuning/finetune.py
"""Two-phase TS1 finetuning sweep: hyperparameter selection then multi-seed runs.

The harness lives in core.sweep; this only picks the TS1 tasks and its train entry.

Knobs (all Hydra, all optional): recording_id (one, a list, or unset for every eval
recording), tasks (a subset of the suite's), sweep.<key> (one grid for every task),
task_sweep.<task>.<key> (a grid for that task only), task_overrides.<task>.<key>
(e.g. a task-specific ckpt.load_from), eval_seeds, ray.gpu, ray.cpu.
"""

import os

os.environ["RAY_ACCEL_ENV_VAR_OVERRIDE_ON_ZERO"] = "0"

import hydra
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, open_dict

from core.launch import run, train
from core.sweep import run_two_phase_sweep
from ibl_bwb_eval.tasks import SUITE_TASKS


@hydra.main(version_base="1.3", config_path="../../configs", config_name="train.yaml")
def main(cfg: DictConfig) -> None:
    with open_dict(cfg):
        cfg.hydra_choices = HydraConfig.get().runtime.choices
        cfg.disable_pbar = True

    run_two_phase_sweep(cfg, train, list(SUITE_TASKS["ts1"]))


if __name__ == "__main__":
    run(main)
