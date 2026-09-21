"""Tests for the two-phase sweep harness in ``core.sweep``.

Covers the pure planning logic: grid expansion, per-task grids, task selection, the
phase-2 project name, and up-front checkpoint validation. The Ray fan-out and the
training itself are not exercised here.
"""

import pytest
from omegaconf import OmegaConf

from core.sweep import (
    expand_grid,
    grid_for_task,
    resolve_tasks,
    seeds_project,
    task_overrides,
    validate_checkpoints,
)

TASKS = ["co_smoothing", "forecasting"]


def test_expand_grid_empty_is_one_empty_config():
    assert expand_grid({}) == [{}]


def test_expand_grid_product():
    assert expand_grid({"a": [1, 2], "b": [3, 4]}) == [
        {"a": 1, "b": 3},
        {"a": 1, "b": 4},
        {"a": 2, "b": 3},
        {"a": 2, "b": 4},
    ]


def test_expand_grid_scalar_is_a_one_value_axis():
    assert expand_grid({"base_lr": 1e-4}) == [{"base_lr": 1e-4}]


def test_single_value_grid_survives_expansion():
    """A one-config grid must reach the run, not be collapsed into no overrides."""
    assert expand_grid({"base_lr": [1e-4]}) == [{"base_lr": 1e-4}]


def test_grid_for_task_prefers_the_task_grid():
    cfg = OmegaConf.create(
        {
            "sweep": {"base_lr": [1e-4, 1e-3]},
            "task_sweep": {"licking_rate": {"base_lr": [3e-5]}},
        }
    )
    assert grid_for_task(cfg, "choice") == [{"base_lr": 1e-4}, {"base_lr": 1e-3}]
    assert grid_for_task(cfg, "licking_rate") == [{"base_lr": 3e-5}]


def test_grid_for_task_without_any_sweep():
    assert grid_for_task(OmegaConf.create({}), "choice") == [{}]


@pytest.mark.parametrize(
    ("tasks", "expected"),
    [
        (None, TASKS),
        ("co_smoothing", ["co_smoothing"]),
        ("co_smoothing, forecasting", TASKS),
        (["forecasting"], ["forecasting"]),
    ],
)
def test_resolve_tasks(tasks, expected):
    cfg = OmegaConf.create({} if tasks is None else {"tasks": tasks})
    assert resolve_tasks(cfg, TASKS) == expected


def test_resolve_tasks_rejects_unknown():
    with pytest.raises(ValueError, match="not valid tasks"):
        resolve_tasks(OmegaConf.create({"tasks": ["bogus"]}), TASKS)


def test_task_overrides_are_flattened_to_dotted_keys():
    cfg = OmegaConf.create({"task_overrides": {"choice": {"ckpt": {"load_from": "a/best.pt"}}}})
    assert task_overrides(cfg, "choice") == {"ckpt.load_from": "a/best.pt"}
    assert task_overrides(cfg, "reward") == {}


def test_seeds_project_defaults_to_hyphen_suffix():
    cfg = OmegaConf.create({"wandb": {"mode": "online", "project": "ts2-mtm-ft"}})
    assert seeds_project(cfg) == "ts2-mtm-ft-seeds"


def test_seeds_project_honors_the_explicit_knob():
    cfg = OmegaConf.create(
        {"wandb": {"mode": "online", "project": "p"}, "wandb_seeds_project": "chosen"}
    )
    assert seeds_project(cfg) == "chosen"


def test_seeds_project_is_none_when_wandb_is_off():
    cfg = OmegaConf.create({"wandb": {"mode": "disabled", "project": "p"}})
    assert seeds_project(cfg) is None


def test_validate_checkpoints_accepts_a_path_under_ckpt_dir(tmp_path):
    (tmp_path / "run").mkdir()
    (tmp_path / "run" / "best.pt").touch()
    cfg = OmegaConf.create({"ckpt": {"dir": str(tmp_path), "load_from": "run/best.pt"}})
    validate_checkpoints(cfg, TASKS)


def test_validate_checkpoints_reports_a_missing_load_from():
    cfg = OmegaConf.create({"ckpt": {"dir": "ckpt/", "load_from": None}})
    with pytest.raises(ValueError, match=r"no ckpt\.load_from"):
        validate_checkpoints(cfg, TASKS)


def test_validate_checkpoints_reports_a_nonexistent_file():
    cfg = OmegaConf.create({"ckpt": {"dir": "ckpt/", "load_from": "nope/best.pt"}})
    with pytest.raises(FileNotFoundError, match="checkpoint not found"):
        validate_checkpoints(cfg, TASKS)


def test_validate_checkpoints_uses_the_per_task_checkpoint(tmp_path):
    (tmp_path / "a.pt").touch()
    cfg = OmegaConf.create(
        {
            "ckpt": {"dir": str(tmp_path), "load_from": None},
            "task_overrides": {t: {"ckpt": {"load_from": "a.pt"}} for t in TASKS},
        }
    )
    validate_checkpoints(cfg, TASKS)


# ---------------------------------------------------------------------------
# Failure isolation. These run a real (local, CPU-only) Ray fan-out over a fake
# train_fn, because the isolation logic is in the wait loop and nowhere else.
# ---------------------------------------------------------------------------

GOOD = "aaaaaaaa-0000-0000-0000-000000000000"
BAD_SEED = "bbbbbbbb-0000-0000-0000-000000000000"
BAD_SWEEP = "cccccccc-0000-0000-0000-000000000000"


def fake_train(cfg, rank, world_size):
    """Fails in the ways the harness is supposed to survive."""
    from core.utils.exceptions import TrainingConstraintsError

    if cfg.recording_id == BAD_SEED and cfg.seed != 42:
        raise TrainingConstraintsError("batch size larger than dataset")
    if cfg.recording_id == BAD_SWEEP and cfg.base_lr == 1e-3:
        return {"best/val/avg": float("nan")}
    return {"best/val/avg": cfg.base_lr * 1000}


def _sweep_cfg(tmp_path, recording_ids):
    (tmp_path / "best.pt").touch()
    return OmegaConf.create(
        {
            "recording_id": recording_ids,
            "tasks": ["co_smoothing"],
            "sweep": {"base_lr": [1e-4, 1e-3]},
            "eval_seeds": [43, 44],
            "base_lr": 1e-3,
            "val": {"minimize": False},
            "wandb": {"mode": "disabled", "project": "test"},
            "ckpt": {"dir": str(tmp_path), "load_from": "best.pt"},
            "ray": {"gpu": 0, "cpu": 0.5},
            "test": {"enable": True},
            "save_preds": {"enable": True},
        }
    )


@pytest.fixture
def _ray_down():
    """Ray workers start with a bare sys.path, so hand them this file's directory."""
    import os
    from pathlib import Path

    previous = os.environ.get("PYTHONPATH", "")
    os.environ["PYTHONPATH"] = os.pathsep.join(filter(None, [str(Path(__file__).parent), previous]))
    yield
    import ray

    ray.shutdown()
    os.environ["PYTHONPATH"] = previous


def test_a_nonfinite_sweep_config_is_pruned_not_fatal(tmp_path, _ray_down):
    from core.sweep import run_two_phase_sweep

    df = run_two_phase_sweep(_sweep_cfg(tmp_path, [BAD_SWEEP]), fake_train, ["co_smoothing"])
    seeds = df[df["phase"] == "seed"]
    assert len(seeds) == 2, "the pair must still get its seed runs"
    # 1e-3 returned nan, so the surviving 1e-4 has to be the selection.
    assert all(o == {"base_lr": 1e-4} for o in seeds["overrides"])


def test_one_dead_pair_does_not_stop_the_others(tmp_path, capsys, _ray_down):
    from core.sweep import run_two_phase_sweep

    cfg = _sweep_cfg(tmp_path, [GOOD, BAD_SEED])
    with pytest.raises(SystemExit, match="1 of 2"):
        run_two_phase_sweep(cfg, fake_train, ["co_smoothing"])

    out = capsys.readouterr().out
    # The surviving pair is still selected on and still reported, and the reported n
    # counts only its seeds: a pair with a ragged seed set must not enter the mean.
    assert "Selected hyperparameters" in out and "Seed runs" in out
    # One failure, not two: abandoning the pair cancels its other queued seed rather
    # than spending a slot to watch it fail the same way.
    assert "Failed jobs (1)" in out


def test_the_sweep_phase_does_not_touch_test_or_predictions(tmp_path, _ray_down):
    """Phase 1 selects on val, so it must not run test or write predictions."""
    from core.sweep import run_two_phase_sweep

    cfg = _sweep_cfg(tmp_path, [GOOD])
    df = run_two_phase_sweep(cfg, record_flags, ["co_smoothing"])
    sweep, seed = df[df["phase"] == "sweep"], df[df["phase"] == "seed"]
    assert not sweep["test_enable"].any() and not sweep["save_preds_enable"].any()
    assert seed["test_enable"].all() and seed["save_preds_enable"].all()


def record_flags(cfg, rank, world_size):
    return {
        "best/val/avg": cfg.base_lr * 1000,
        "test_enable": bool(cfg.test.enable),
        "save_preds_enable": bool(cfg.save_preds.enable),
    }
