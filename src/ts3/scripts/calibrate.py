# src/ts3/scripts/calibrate.py
"""Ray sweep for TS3 calibration: every eval recording x every seed, one Ray task each.

Unlike TS1/TS2's finetuning sweep (``core.sweep.run_two_phase_sweep``), there is no
hyperparameter grid and no per-task axis here: a calibrate trainer's hyperparameters are
fixed in its own config (``ts3/models/transductive/<model>/configs/trainer/``), and there
is exactly one calibration per (recording, seed), not per (recording, task, seed).

Configurable parameters (all Hydra, all optional):
- recording_id (one, a list, or unset for every eval recording)
- eval_seeds (default the benchmark's 5)
- ray.gpu, ray.cpu (for Ray resource allocation)
"""

import copy
import logging
import os

import hydra
import pandas as pd
import ray
import torch
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf, open_dict
from rich.console import Console
from rich.table import Table

from core.launch import run, train
from core.sweep import DEFAULT_RAY_CPU, DEFAULT_RAY_GPU
from core.utils.exceptions import TrainingConstraintsError
from ibl_bwb_eval.protocol import EVAL_RECORDING_IDS, EVAL_SEEDS

log = logging.getLogger(__name__)


def resolve_recording_ids(cfg: DictConfig) -> list[str]:
    recording_ids = OmegaConf.select(cfg, "recording_id")
    if recording_ids is None:
        return pd.read_csv(EVAL_RECORDING_IDS, header=None)[0].tolist()
    if isinstance(recording_ids, str):
        return [recording_ids]
    return list(recording_ids)


def _classify(exc: BaseException) -> str:
    """Short label for the failure table."""
    if isinstance(exc, TrainingConstraintsError):
        return "constraints"
    if isinstance(exc, torch.cuda.OutOfMemoryError):
        return "oom"
    return type(exc).__name__


@ray.remote(num_gpus=DEFAULT_RAY_GPU, num_cpus=DEFAULT_RAY_CPU)
def run_job(cfg: DictConfig, recording_id: str, seed: int) -> dict:
    cfg = copy.deepcopy(cfg)
    with open_dict(cfg):
        cfg.recording_ids = [recording_id]
        cfg.seed = seed

    result = train(cfg, 0, 1)
    if not result:
        raise RuntimeError(f"{recording_id[:8]} seed {seed} returned no metrics")
    return {"recording_id": recording_id, "seed": seed, **result}


def display_failures(failures: list[dict]) -> None:
    table = Table(title=f"Failed jobs ({len(failures)})", show_header=True)
    for col in ("recording_id", "seed", "kind"):
        table.add_column(col, style="cyan")
    table.add_column("error", style="red", overflow="fold")
    for fail in failures:
        table.add_row(
            fail["recording_id"][:8], str(fail["seed"]), fail["kind"], fail["error"][:160]
        )
    Console().print(table)


def run_calibration_sweep(cfg: DictConfig) -> pd.DataFrame:
    """Calibrate every (recording, seed) pair and report. Exits non-zero if any failed."""
    recording_ids = resolve_recording_ids(cfg)
    seeds = OmegaConf.select(cfg, "eval_seeds") or EVAL_SEEDS
    n_jobs = len(recording_ids) * len(seeds)
    log.info(f"{len(recording_ids)} recording(s) x {len(seeds)} seed(s) = {n_jobs} jobs")

    resources = {}
    for k in ("gpu", "cpu"):
        v = OmegaConf.select(cfg, f"ray.{k}")
        if v is not None:
            resources[f"num_{k}s"] = v
    f = run_job.options(**resources) if resources else run_job
    log.info(f"ray resources per job: {resources or 'default'}")

    ray.init(
        ignore_reinit_error=True,
        log_to_driver=True,
        num_gpus=torch.cuda.device_count(),
        num_cpus=len(os.sched_getaffinity(0)),
    )
    log.info(f"ray cluster resources: {ray.cluster_resources()}")

    n_gpu = ray.cluster_resources().get("GPU", 0)
    need = resources.get("num_gpus", DEFAULT_RAY_GPU)
    if n_gpu < need:
        raise RuntimeError(
            f"ray sees {n_gpu} GPUs, each job needs {need}; jobs would idle until walltime"
        )

    jobs = {f.remote(cfg, rid, seed): (rid, seed) for rid in recording_ids for seed in seeds}
    pending = list(jobs)
    results: list[dict] = []
    failures: list[dict] = []

    try:
        while pending:
            done, pending = ray.wait(pending)
            rid, seed = jobs.pop(done[0])
            tag = f"[{rid[:8]} seed {seed}]"
            try:
                results.append(ray.get(done[0]))
            except Exception as exc:
                kind = _classify(exc)
                failures.append(
                    {"recording_id": rid, "seed": seed, "kind": kind, "error": str(exc)}
                )
                log.error(f"{tag} failed ({kind}): {exc}")

        log.info(f"{len(results)}/{n_jobs} jobs done, {len(failures)} failed")
    finally:
        log.info("shutting down Ray...")
        ray.shutdown()

    if failures:
        display_failures(failures)
    if not results:
        raise SystemExit(f"all {n_jobs} jobs failed")
    return pd.DataFrame(results)


@hydra.main(version_base="1.3", config_path="../configs", config_name="train.yaml")
def main(cfg: DictConfig) -> None:
    with open_dict(cfg):
        cfg.hydra_choices = HydraConfig.get().runtime.choices
        cfg.disable_pbar = True

    run_calibration_sweep(cfg)


if __name__ == "__main__":
    run(main)
