# src/core/sweep.py
"""Two-phase sweep harness: per-(recording, task) hparam selection, then multi-seed runs.

Phase 1 fans a grid over ``SWEEP_SEED`` and selects on ``SELECTION_METRIC``, which is a
validation metric: phase-1 jobs run with test and prediction dumping off. Phase 2 reruns
each winner over ``eval_seeds`` in its own W&B project. The TS1 and TS2 finetuning entry
points supply only their task list and their ``train``.
"""

import copy
import itertools
import logging
import os
from collections import defaultdict
from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import ray
import torch
from omegaconf import DictConfig, OmegaConf, open_dict
from rich.console import Console
from rich.table import Table

from core.utils.exceptions import TrainingConstraintsError
from core.utils.util import expand_path
from ibl_bwb_eval.protocol import EVAL_RECORDING_IDS, EVAL_SEEDS, SELECTION_METRIC, SWEEP_SEED

log = logging.getLogger(__name__)

DEFAULT_RAY_GPU = 0.125
DEFAULT_RAY_CPU = 3.5


@dataclass
class Job:
    recording_id: str
    task: str
    seed: int
    phase: str  # "sweep" or "seed"
    overrides: dict = field(default_factory=dict)


def _set_nested(cfg: DictConfig, key: str, value) -> None:
    """Set a possibly-dotted key in cfg, creating intermediate nodes as needed."""
    OmegaConf.update(cfg, key, value, force_add=True)


def _flatten(d, prefix="") -> dict:
    """Flatten a nested config to dotted keys."""
    out = {}
    for k, v in d.items():
        key = f"{prefix}.{k}" if prefix else k
        if isinstance(v, DictConfig | dict):
            out.update(_flatten(v, key))
        else:
            out[key] = v
    return out


def _as_axis(v) -> list:
    """A scalar is a one-value axis, not an iterable of values."""
    return list(v) if isinstance(v, list | tuple) else [v]


def expand_grid(sweep: dict) -> list[dict]:
    """Expand a mapping of axes to the list of every combination across them.

    ``{'a': [1,2], 'b': [3,4]}`` -> ``[{'a':1,'b':3}, {'a':1,'b':4}, ...]``
    """
    if not sweep:
        return [{}]
    keys = list(sweep.keys())
    values = [_as_axis(sweep[k]) for k in keys]
    return [dict(zip(keys, combo, strict=True)) for combo in itertools.product(*values)]


def grid_for_task(cfg: DictConfig, task: str) -> list[dict]:
    """Grid for one task: ``task_sweep.<task>`` if present, else the global ``sweep``.

    Per-task grids let one invocation cover behaviors whose usable lr ranges differ,
    instead of forcing the widest grid on all of them.
    """
    node = OmegaConf.select(cfg, f"task_sweep.{task}")
    if node is None:
        node = OmegaConf.select(cfg, "sweep")
    sweep = OmegaConf.to_container(node, resolve=True) if node else {}
    return expand_grid(sweep)


def task_overrides(cfg: DictConfig, task: str) -> dict:
    """Per-task config overrides, e.g. a task-specific ``ckpt.load_from``."""
    node = OmegaConf.select(cfg, f"task_overrides.{task}")
    return _flatten(node) if node else {}


def fmt_overrides(o: dict) -> str:
    if not o:
        return "(none)"
    return ", ".join(f"{k}={_fmt_val(v)}" for k, v in sorted(o.items()))


def _fmt_val(v) -> str:
    if isinstance(v, float):
        return f"{v:.2e}" if (abs(v) < 1e-2 or abs(v) >= 1e4) else f"{v:g}"
    return str(v)


def display_results(df: pd.DataFrame, title: str) -> None:
    skip = {"recording_id", "task", "seed", "phase", "overrides"}
    metrics = [c for c in df.columns if c not in skip]
    table = Table(title=title, show_header=True)
    table.add_column("Metric", style="cyan")
    table.add_column("Mean", style="green", justify="right")
    table.add_column("Std", style="magenta", justify="right")
    table.add_column("SEM", style="yellow", justify="right")
    table.add_column("n", justify="right")
    for m in metrics:
        vals = df[m].dropna().values
        n = len(vals)
        if n == 0:
            continue
        mean = np.mean(vals)
        std = np.std(vals, ddof=1) if n > 1 else 0.0
        sem = std / np.sqrt(n) if n > 1 else 0.0
        table.add_row(m, f"{mean:.4f}", f"{std:.4f}", f"{sem:.4f}", str(n))
    Console().print(table)


def display_sweep_winners(best: dict, sweep_results: dict, minimize: bool) -> None:
    goal = "min" if minimize else "max"
    table = Table(
        title=f"Selected hyperparameters per (recording_id, task), {goal} {SELECTION_METRIC}",
        show_header=True,
    )
    table.add_column("recording_id", style="cyan")
    table.add_column("task", style="cyan")
    table.add_column("best overrides", style="green")
    table.add_column(SELECTION_METRIC, style="yellow", justify="right")
    table.add_column("margin", style="yellow", justify="right")
    table.add_column("all configs", style="white")
    for pair, winner_overrides in best.items():
        rid, task = pair
        results = sweep_results[pair]
        if not results:
            continue
        winner_metric = next(m for o, m in results if o == winner_overrides)
        ranked = sorted(results, key=lambda x: x[1], reverse=not minimize)
        # Gap to the runner-up, so a coin-flip selection is visible in the log.
        margin = abs(ranked[0][1] - ranked[1][1]) if len(ranked) > 1 else float("nan")
        table.add_row(
            rid[:8],
            task,
            fmt_overrides(winner_overrides),
            f"{winner_metric:.4f}",
            f"{margin:.4f}",
            " | ".join(f"{fmt_overrides(o)}={m:.3f}" for o, m in ranked),
        )
    Console().print(table)


def resolve_recording_ids(cfg: DictConfig) -> list[str]:
    recording_ids = OmegaConf.select(cfg, "recording_id")
    if recording_ids is None:
        return pd.read_csv(EVAL_RECORDING_IDS, header=None)[0].tolist()
    if isinstance(recording_ids, str):
        return [recording_ids]
    return list(recording_ids)


def resolve_tasks(cfg: DictConfig, all_tasks: list[str]) -> list[str]:
    """Tasks to run, unset meaning all of them.

    A subset is how a model whose checkpoints are task-specific, and so do not exist for
    every behavior, gets evaluated.
    """
    tasks = OmegaConf.select(cfg, "tasks")
    if tasks is None:
        return list(all_tasks)
    if isinstance(tasks, str):
        tasks = [t.strip() for t in tasks.split(",") if t.strip()]
    else:
        tasks = list(tasks)
    invalid = [t for t in tasks if t not in all_tasks]
    if invalid:
        raise ValueError(f"{invalid} are not valid tasks; valid: {all_tasks}")
    return tasks


def seeds_project(cfg: DictConfig) -> str | None:
    """W&B project for phase 2, keeping the seed runs out of the sweep's project."""
    if OmegaConf.select(cfg, "wandb.mode") == "disabled":
        return None
    project = OmegaConf.select(cfg, "wandb.project")
    return OmegaConf.select(cfg, "wandb_seeds_project") or (project and f"{project}-seeds")


def validate_checkpoints(cfg: DictConfig, tasks: list[str]) -> None:
    """Resolve every task's checkpoint before a single job is queued.

    Both failures used to surface inside a Ray worker, hours in and after the whole grid
    was already submitted. Mirrors the two candidates ``BaseTrainer._load_ckpt`` tries.
    """
    missing, not_found = [], []
    for task in tasks:
        load_from = task_overrides(cfg, task).get("ckpt.load_from") or OmegaConf.select(
            cfg, "ckpt.load_from"
        )
        if load_from is None:
            missing.append(task)
            continue
        candidates = [expand_path(load_from)]
        ckpt_dir = OmegaConf.select(cfg, "ckpt.dir")
        if ckpt_dir:
            candidates.append(expand_path(ckpt_dir) / load_from)
        if not any(p.exists() for p in candidates):
            not_found.append(f"{task}: {' or '.join(str(p) for p in candidates)}")
    if missing:
        raise ValueError(
            f"no ckpt.load_from for {missing}; set it globally or per task under "
            "task_overrides.<task>.ckpt.load_from"
        )
    if not_found:
        raise FileNotFoundError("checkpoint not found for " + "; ".join(not_found))


@ray.remote(num_gpus=DEFAULT_RAY_GPU, num_cpus=DEFAULT_RAY_CPU)
def run_job(train_fn: Callable, cfg: DictConfig, job: Job) -> dict:
    cfg = copy.deepcopy(cfg)
    with open_dict(cfg):
        cfg.recording_id = job.recording_id
        cfg.task = job.task
        cfg.seed = job.seed

        for k, v in job.overrides.items():
            _set_nested(cfg, k, v)
        # Applied after the grid so a per-task value (the checkpoint) wins over it.
        # To sweep an axis per task use task_sweep, not this.
        for k, v in task_overrides(cfg, job.task).items():
            _set_nested(cfg, k, v)

        if job.phase == "sweep":
            # Selection is on val, so phase 1 has no business touching test.
            _set_nested(cfg, "save_preds.enable", False)
            _set_nested(cfg, "test.enable", False)

    result = train_fn(cfg, 0, 1)
    if not result:
        raise RuntimeError(f"{job.recording_id[:8]}/{job.task} seed {job.seed} returned no metrics")
    return {
        "recording_id": job.recording_id,
        "task": job.task,
        "seed": job.seed,
        "phase": job.phase,
        "overrides": dict(job.overrides),
        **result,
    }


def _classify(exc: BaseException) -> str:
    """Short label for the failure table."""
    if isinstance(exc, TrainingConstraintsError):
        return "constraints"
    if isinstance(exc, torch.cuda.OutOfMemoryError):
        return "oom"
    return type(exc).__name__


def display_failures(failures: list[dict], dead: set) -> None:
    table = Table(title=f"Failed jobs ({len(failures)})", show_header=True)
    for col in ("recording_id", "task", "seed", "phase", "kind"):
        table.add_column(col, style="cyan")
    table.add_column("error", style="red", overflow="fold")
    for fail in failures:
        table.add_row(
            fail["recording_id"][:8],
            fail["task"],
            str(fail["seed"]),
            fail["phase"],
            fail["kind"],
            fail["error"][:160],
        )
    Console().print(table)
    if dead:
        pairs = ", ".join(f"{rid[:8]}/{task}" for rid, task in sorted(dead))
        log.error(f"{len(dead)} (recording, task) pair(s) produced no result: {pairs}")


def run_two_phase_sweep(cfg: DictConfig, train_fn: Callable, all_tasks: list[str]) -> pd.DataFrame:
    """Run both phases and report. Exits non-zero if any pair produced no result.

    Failures are isolated per (recording_id, task): a crashed or non-finite sweep config
    drops out of that pair's candidate set, and only a pair whose every config failed, or
    one that fails during phase 2, is abandoned. Every other pair runs to completion.
    """
    recording_ids = resolve_recording_ids(cfg)
    tasks = resolve_tasks(cfg, all_tasks)
    seeds = OmegaConf.select(cfg, "eval_seeds") or EVAL_SEEDS
    minimize = bool(OmegaConf.select(cfg, "val.minimize"))
    validate_checkpoints(cfg, tasks)

    pairs = [(rid, task) for rid in recording_ids for task in tasks]
    grids = {task: grid_for_task(cfg, task) for task in tasks}
    do_sweep = any(len(g) > 1 for g in grids.values())
    n_seed_jobs = len(pairs) * len(seeds)

    log.info(f"{len(recording_ids)} recording(s) x tasks {tasks}")
    for task, g in grids.items():
        log.info(f"  {task}: {len(g)} config(s): {' | '.join(fmt_overrides(o) for o in g)}")
    if do_sweep:
        n_sweep_jobs = sum(len(grids[task]) for _, task in pairs)
        goal = "min" if minimize else "max"
        log.info(f"  phase 1 (hparam sweep): {n_sweep_jobs} jobs, {goal} {SELECTION_METRIC}")
        log.info(f"  phase 2 (seeds):        {n_seed_jobs} jobs")
    else:
        log.info(f"no sweep; {len(pairs)} pairs x {len(seeds)} seeds = {n_seed_jobs} jobs")

    resources = {}
    for k in ("gpu", "cpu"):
        v = OmegaConf.select(cfg, f"ray.{k}")
        if v is not None:
            resources[f"num_{k}s"] = v
    f = run_job.options(**resources) if resources else run_job
    log.info(f"ray resources per job: {resources or 'default'}")

    # Phase 2 gets its own project so the seed runs are not mixed in with the grid.
    seed_cfg = cfg
    project = seeds_project(cfg)
    if project and project != OmegaConf.select(cfg, "wandb.project"):
        seed_cfg = copy.deepcopy(cfg)
        with open_dict(seed_cfg):
            _set_nested(seed_cfg, "wandb.project", project)
        log.info(f"phase 2 W&B project: {project}")

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

    sweep_pending: dict = defaultdict(int)
    sweep_results: dict = defaultdict(list)  # pair -> [(overrides_dict, metric), ...]
    best: dict = {}  # pair -> winning overrides_dict
    all_results: list = []
    pending: list = []
    jobs: dict = {}  # ObjectRef -> Job, so a failure can name what failed
    failures: list = []
    dead: set = set()  # pairs abandoned, excluded from the reported means

    def submit(cfg_, job):
        ref = f.remote(train_fn, cfg_, job)
        jobs[ref] = job
        pending.append(ref)

    def submit_seed_jobs(pair, overrides):
        for seed in seeds:
            submit(seed_cfg, Job(pair[0], pair[1], seed, "seed", dict(overrides)))

    def record_failure(job, kind, error):
        failures.append(
            {
                "recording_id": job.recording_id,
                "task": job.task,
                "seed": job.seed,
                "phase": job.phase,
                "kind": kind,
                "error": str(error),
            }
        )

    def abandon(pair, reason):
        """Stop waiting on a pair's queued jobs and free their slots for the rest."""
        nonlocal pending
        dead.add(pair)
        keep = []
        for ref in pending:
            if (jobs[ref].recording_id, jobs[ref].task) == pair:
                ray.cancel(ref)
                jobs.pop(ref)
            else:
                keep.append(ref)
        pending = keep
        log.error(f"[{pair[0][:8]}/{pair[1]}] abandoned: {reason}")

    def finish_sweep(pair):
        """Select among the configs that survived, or abandon the pair if none did."""
        if not sweep_results[pair]:
            abandon(pair, f"all {len(grids[pair[1]])} sweep configs failed")
            return
        pick = min if minimize else max
        chosen_overrides, chosen_metric = pick(sweep_results[pair], key=lambda x: x[1])
        best[pair] = chosen_overrides
        log.info(
            f"[{pair[0][:8]}/{pair[1]}] sweep done "
            f"({len(sweep_results[pair])}/{len(grids[pair[1]])} configs usable): "
            f"chose {fmt_overrides(chosen_overrides)} "
            f"({SELECTION_METRIC}={chosen_metric:.4f}); "
            f"queueing {len(seeds)} seed jobs"
        )
        submit_seed_jobs(pair, chosen_overrides)

    try:
        if do_sweep:
            for pair in pairs:
                for overrides in grids[pair[1]]:
                    submit(cfg, Job(pair[0], pair[1], SWEEP_SEED, "sweep", overrides))
                    sweep_pending[pair] += 1
        else:
            # A one-config grid is still a config to apply, not an empty override set.
            for pair in pairs:
                best[pair] = grids[pair[1]][0]
                submit_seed_jobs(pair, best[pair])

        while pending:
            done, pending = ray.wait(pending)
            job = jobs.pop(done[0])
            pair = (job.recording_id, job.task)
            tag = f"[{pair[0][:8]}/{pair[1]}] {job.phase} seed {job.seed}"

            try:
                result = ray.get(done[0])
            except Exception as exc:
                kind = _classify(exc)
                record_failure(job, kind, exc)
                log.error(f"{tag} failed ({kind}): {exc}")
                if job.phase == "sweep":
                    # A config that crashes is out of the running, not a dead pair.
                    sweep_pending[pair] -= 1
                    if sweep_pending[pair] == 0:
                        finish_sweep(pair)
                elif pair not in dead:
                    abandon(pair, f"seed {job.seed} failed ({kind})")
                continue

            all_results.append(result)

            if job.phase == "sweep":
                metric = result.get(SELECTION_METRIC)
                if metric is None or not np.isfinite(metric):
                    # Diverging at the top of a grid is expected; drop the config.
                    record_failure(job, "nonfinite", f"{SELECTION_METRIC}={metric}")
                    log.warning(
                        f"{tag} {fmt_overrides(job.overrides)}: {SELECTION_METRIC}={metric}"
                    )
                else:
                    sweep_results[pair].append((result["overrides"], metric))
                sweep_pending[pair] -= 1
                if sweep_pending[pair] == 0:
                    finish_sweep(pair)

        log.info(f"{len(all_results)} jobs done, {len(failures)} failed, {len(dead)} pairs dropped")
        df = pd.DataFrame(all_results)

        if do_sweep:
            display_sweep_winners(best, sweep_results, minimize)
        if len(df):
            shown = df[df["phase"] == "seed"] if do_sweep else df
            # Ragged seed counts would make the mean incomparable across pairs.
            shown = shown[
                [
                    (rid, task) not in dead
                    for rid, task in zip(shown["recording_id"], shown["task"], strict=True)
                ]
            ]
            if len(shown):
                display_results(shown, "Seed runs (mean ± std ± sem)")

    finally:
        log.info("shutting down Ray...")
        ray.shutdown()

    if failures:
        display_failures(failures, dead)
    if dead:
        # Pruned configs are normal; a pair with no result at all is not.
        raise SystemExit(f"{len(dead)} of {len(pairs)} (recording, task) pairs produced no result")
    return df
