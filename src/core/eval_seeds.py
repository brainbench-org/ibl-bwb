import os

os.environ["RAY_ACCEL_ENV_VAR_OVERRIDE_ON_ZERO"] = "0"

import argparse
import copy
import json

import numpy as np
import pandas as pd
import ray
import wandb
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf, open_dict
from rich.console import Console
from rich.table import Table

from core.launch import train
from core.utils.logger import get_cli_logger
from ibl_bwb_eval.protocol import EVAL_SEEDS

logger = get_cli_logger()


def summarize_seeds(df: pd.DataFrame) -> dict[str, tuple[float, float, float, int]]:
    """Per-metric ``(mean, std, sem, n)`` over ``df``, which holds one row per seed."""
    stats = {}
    for metric in (col for col in df.columns if col != "seed"):
        values = df[metric].dropna().values
        n = len(values)
        if n == 0:
            continue
        std = float(np.std(values, ddof=1)) if n > 1 else 0.0
        sem = std / np.sqrt(n) if n > 1 else 0.0
        stats[metric] = (float(np.mean(values)), std, float(sem), n)
    return stats


def print_seed_summary(df: pd.DataFrame) -> None:
    """The standard multi-seed results table, shared by both seed runners."""
    table = Table(
        title="Final results over n seeds (mean | ± std | ± sem | n)",
        show_header=True,
    )
    table.add_column("Metric", style="cyan", justify="left")
    table.add_column("Mean", style="green", justify="right")
    table.add_column("Std", style="magenta", justify="right")
    table.add_column("SEM", style="yellow", justify="right")
    table.add_column("n", style="white", justify="right")
    for metric, (mean, std, sem, n) in summarize_seeds(df).items():
        table.add_row(metric, f"{mean:.4f}", f"{std:.4f}", f"{sem:.4f}", str(n))
    Console().print(table)


def run_seeds(cfg: DictConfig, seeds: list[int] | None = None) -> pd.DataFrame:
    """Train ``cfg`` once per eval seed, in this process, and summarize the results.

    Each seed trains from its own deep copy, so nothing a trainer writes into ``cfg``
    can reach the next seed. Suites call this from their ``train_seeds.py`` shim; use
    :func:`run_final_eval` instead to fan the same thing out over Ray.
    """
    seeds = seeds or OmegaConf.select(cfg, "eval_seeds") or EVAL_SEEDS
    with open_dict(cfg):
        cfg.hydra_choices = HydraConfig.get().runtime.choices

    results = []
    for seed in seeds:
        cfg_seed = copy.deepcopy(cfg)
        with open_dict(cfg_seed):
            cfg_seed.seed = seed
        results.append(train(cfg_seed, 0, 1))

    df = pd.DataFrame(results)
    if len(results) > 1:
        print_seed_summary(df)
    return df


@ray.remote(num_gpus=1, num_cpus=1)
def final_eval(cfg: DictConfig, seed: int):
    cfg.seed = seed
    return train(cfg, 0, 1)


def run_final_eval(cfg: DictConfig, display_results: bool = True):
    resources = {}
    for k in ("gpu", "cpu"):
        val = OmegaConf.select(cfg, f"ray.{k}")
        if val is not None:
            resources[f"num_{k}s"] = val
    f = final_eval.options(**resources) if resources else final_eval

    logger.info("=" * 23 + " FINAL EVAL: MULTI-SEED " + "=" * 23)
    logger.info(f"Evaluating over seeds: {cfg.eval_seeds}")
    logger.info(f"Resources: {resources}")
    logger.info("=" * 70 + "\n")

    # required configs
    if cfg.wandb.mode != "disabled":
        cfg.wandb.project = cfg.wandb_seeds_project or cfg.wandb.project
    assert hasattr(cfg, "eval_seeds"), (
        "eval_seeds must be set in config to evaluate over multiple seeds"
    )

    # run final evaluation
    results = []
    for seed in cfg.eval_seeds:
        results.append(f.remote(cfg, seed))

    final_results = []
    while results:
        done_id, results = ray.wait(results)
        result = ray.get(done_id[0])
        final_results.append(result)
    df = pd.DataFrame(final_results)

    if display_results:
        logger.info("Final Results (mean ± std ± sem over seeds):")
        print_seed_summary(df)

    return df


def evaluate_wandb_run(run: wandb.Run, args: argparse.Namespace, extra: list[str]):
    cfg = OmegaConf.create(run.config)

    overrides = OmegaConf.from_dotlist(extra)
    cfg = OmegaConf.merge(cfg, overrides)
    cfg.disable_pbar = True

    # parse seeds
    if "[" in args.eval_seeds and "]" in args.eval_seeds:
        eval_seeds = [int(s) for s in args.eval_seeds.strip("[]").split(",")]
    elif "," in args.eval_seeds:
        eval_seeds = [int(s) for s in args.eval_seeds.split(",")]
    else:
        eval_seeds = [int(args.eval_seeds)]
    cfg.eval_seeds = eval_seeds

    if args.gpu is not None:
        OmegaConf.update(cfg, "ray.gpu", args.gpu, force_add=True)
    if args.cpu is not None:
        OmegaConf.update(cfg, "ray.cpu", args.cpu, force_add=True)
    if args.seeds_project is None:
        OmegaConf.update(cfg, "wandb_seeds_project", cfg.wandb.project, force_add=True)
    else:
        OmegaConf.update(cfg, "wandb_seeds_project", args.seeds_project, force_add=True)

    try:
        run_final_eval(cfg, display_results=True)
    finally:
        logger.info("Shutting down Ray...")
        ray.shutdown()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_project", type=str, required=True)
    parser.add_argument("--seeds_project", type=str, default=None)
    parser.add_argument("--run_id", type=str, required=False)
    parser.add_argument("--wandb_filters", type=str, required=False)
    parser.add_argument("--eval_seeds", type=str, required=True)
    parser.add_argument("--entity", type=str, default=os.environ.get("WANDB_ENTITY"))
    parser.add_argument("--cpu", type=float, default=None)
    parser.add_argument("--gpu", type=float, default=None)
    args, extra = parser.parse_known_args()

    assert args.run_id is not None or args.wandb_filters is not None, (
        "Either --run_id or --wandb_filters must be provided"
    )

    api = wandb.Api()
    if api.api_key is None:
        raise SystemExit(
            "Reading runs from W&B needs an API key: set WANDB_API_KEY (see .env.example)."
        )
    scope = f"{args.entity}/{args.run_project}" if args.entity else args.run_project
    if args.run_id is not None:
        runs = [api.run(f"{scope}/{args.run_id}")]
    else:
        filters = json.loads(args.wandb_filters) if args.wandb_filters else {}
        runs = api.runs(scope, filters=filters)
    logger.info(f"Found {len(runs)} runs matching filters: {args.wandb_filters}")

    for run in runs:
        evaluate_wandb_run(run, args, extra)


if __name__ == "__main__":
    main()
