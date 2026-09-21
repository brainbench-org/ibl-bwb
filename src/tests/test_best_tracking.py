"""Tests for best-model selection: the direction each trainer selects in, and the
bookkeeping :meth:`BaseTrainer.save_if_best` keeps.

The failure this guards against is silent. A trainer whose ``best_val`` is seeded with
the wrong sign compares against a sentinel it can never beat, so it records no best,
writes no ``best.pt``, and reports nothing unusual; the run simply ends up selecting
last-epoch weights. Direction reaches ``save_if_best`` from two places, config and an
explicit ``reset_best_tracking(minimize=...)``, so the check has to resolve both.
"""

import ast
import importlib
import inspect
from pathlib import Path

import pytest
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

from core.trainer import BaseTrainer

SRC = Path(__file__).parents[1]

# Attributes reset_best_tracking owns. Only best_model and best_metrics are defined for
# every trainer; the rest appear once a trainer opts into tracking a best.
ALWAYS_DEFINED = ("best_model", "best_metrics")
OPT_IN = ("patience", "start_patience", "best_val", "best_epoch", "val_minimize")

# Each trainer that selects a best, with the direction its metric wants. `minimize` is
# the expected answer, and the note says which metric it is read from; keep the two in
# step when a trainer changes what it selects on.
TRAINERS = [
    ("ts1", "mlp", ["task=choice"], False, "readout primary_metric (bacc/r2/poisson_d2)"),
    ("ts2", "ndt", ["task=co_smoothing"], False, "poisson_d2"),
    ("ts3", "lolcat", [], False, "val/f1"),
    ("pretrain", "nemo_pretrain", [], False, "brain-region probe f1_macro"),
    ("pretrain", "neds_pretrain", [], True, "val_loss"),
    ("pretrain", "possm_pretrain", [], False, "readout primary_metric"),
    ("pretrain", "poyo_pretrain", ["task=choice"], False, "readout primary_metric"),
    ("pretrain", "possm_multitask_pretrain", [], False, "avg_metric"),
    ("pretrain", "poyo_plus_multitask_pretrain", [], False, "avg_metric"),
    ("pretrain", "rrr_pretrain", ["task=choice"], False, "readout primary_metric"),
    ("pretrain", "mtm_pretrain", [], True, "val_loss"),
    ("pretrain", "ndt2_pretrain", [], True, "val_loss"),
    ("pretrain", "ndt_stitch_pretrain", [], True, "val_loss"),
    ("pretrain", "my_model_pretrain", [], True, "val_loss (the scaffold)"),
]

IDS = [f"{pkg}/{trainer}" for pkg, trainer, *_ in TRAINERS]


def _compose(pkg: str, trainer: str, overrides: list[str]):
    with initialize_config_dir(config_dir=str(SRC / pkg / "configs"), version_base="1.2"):
        return compose(
            config_name="train.yaml",
            overrides=[f"trainer={trainer}", "data_root=/tmp/x", *overrides],
        )


def _trainer_class(target: str) -> type:
    """Resolve the class a ``trainer._target_`` names, following re-exports."""
    module_path, class_name = target.rsplit(".", 1)
    return getattr(importlib.import_module(module_path), class_name)


def _defining_files(cls: type) -> set[Path]:
    """Every file in the class's MRO below BaseTrainer.

    A trainer often inherits ``val_epoch``, and so its ``save_if_best`` call, from a
    suite-level base, so the call site is rarely in the file ``_target_`` points at.
    """
    return {
        Path(inspect.getfile(klass)) for klass in cls.__mro__ if klass not in (BaseTrainer, object)
    }


def _explicit_minimize(path: Path, class_name: str) -> bool | None:
    """The literal in ``reset_best_tracking(minimize=...)``, if the class passes one.

    A trainer whose direction is fixed in code passes it here rather than through
    config, and that call wins, so config alone is not the effective answer.
    """
    tree = ast.parse(path.read_text())
    cls = next(
        (n for n in ast.walk(tree) if isinstance(n, ast.ClassDef) and n.name == class_name),
        None,
    )
    if cls is None:
        return None
    for node in ast.walk(cls):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        if not (isinstance(func, ast.Attribute) and func.attr == "reset_best_tracking"):
            continue
        for kw in node.keywords:
            if kw.arg == "minimize" and isinstance(kw.value, ast.Constant):
                return bool(kw.value.value)
    return None


def _effective_minimize(pkg, trainer, overrides) -> bool:
    cfg = _compose(pkg, trainer, overrides)
    cls = _trainer_class(cfg.trainer._target_)
    for klass in cls.__mro__:
        if klass in (BaseTrainer, object):
            break
        explicit = _explicit_minimize(Path(inspect.getfile(klass)), klass.__name__)
        if explicit is not None:
            return explicit
    return bool(OmegaConf.select(cfg, "val.minimize", default=False))


# ---------------------------------------------------------------- direction per trainer


@pytest.mark.parametrize(("pkg", "trainer", "overrides", "minimize", "note"), TRAINERS, ids=IDS)
def test_selection_direction_matches_the_metric(pkg, trainer, overrides, minimize, note):
    effective = _effective_minimize(pkg, trainer, overrides)
    assert effective is minimize, (
        f"{pkg}/{trainer} selects on {note}, which wants minimize={minimize}, "
        f"but resolves to minimize={effective}. It would compare against a "
        f"{'+' if effective else '-'}inf sentinel and never record a best."
    )


@pytest.mark.parametrize(("pkg", "trainer", "overrides", "minimize", "note"), TRAINERS, ids=IDS)
def test_seeded_sentinel_can_be_beaten(pkg, trainer, overrides, minimize, note):
    """The seeded ``best_val`` has to lose to a real score, or nothing is ever kept."""
    trainer_obj = _stub(explicit=_effective_minimize(pkg, trainer, overrides))
    assert trainer_obj.save_if_best(0.5) is True, f"{pkg}/{trainer} cannot beat its own sentinel"


def test_every_trainer_that_selects_a_best_is_covered():
    """A new trainer calling save_if_best has to declare its direction here."""
    callers = {
        path.relative_to(SRC)
        for path in SRC.rglob("*.py")
        if "save_if_best(" in (source := path.read_text())
        and "def save_if_best" not in source  # core defines it
        and path.parent.name != "tests"
    }
    covered = set()
    for pkg, trainer, overrides, *_ in TRAINERS:
        cfg = _compose(pkg, trainer, overrides)
        for path in _defining_files(_trainer_class(cfg.trainer._target_)):
            covered.add(path.relative_to(SRC))
    missing = callers - covered
    assert not missing, f"these call save_if_best but declare no direction: {sorted(missing)}"


# ---------------------------------------------------------------- the mechanism


class _Stub(BaseTrainer):
    """The smallest thing that can exercise the tracking methods."""

    def setup(self, ckpt):  # pragma: no cover - never constructed through __init__
        ...

    def train_epoch(self): ...

    def val_epoch(self): ...


def _stub(explicit=None, **val):
    obj = _Stub.__new__(_Stub)
    obj.cfg = OmegaConf.create({"num_epochs": 10, "val": val})
    obj.epoch = 0
    obj.best_model = None
    obj.best_metrics = {}
    obj._best_tracking_initialized = False
    obj.model = _ConstantModel()
    obj.logger = _NullLogger()
    obj._save_best_checkpoint = lambda: None
    obj.reset_best_tracking(minimize=explicit)
    return obj


class _ConstantModel:
    def state_dict(self):
        return {"w": 1.0}


class _NullLogger:
    def info(self, *_a, **_k): ...

    def log_dict(self, *_a, **_k): ...


def test_direction_comes_from_config_and_defaults_to_maximizing():
    assert _stub().best_val == float("-inf")
    assert _stub(minimize=True).best_val == float("inf")
    assert _stub(minimize=False).best_val == float("-inf")


def test_an_explicit_direction_beats_config():
    """The trainers whose direction is fixed in code rely on this argument winning."""
    assert _stub(explicit=False, minimize=True).best_val == float("-inf")
    assert _stub(explicit=True, minimize=False).best_val == float("inf")


def test_a_non_finite_score_is_never_an_improvement():
    """NaN compares False both ways, so an unguarded one would be recorded as best and
    then lose every later comparison, letting every subsequent epoch count as better."""
    t = _stub(minimize=False)
    assert t.save_if_best(float("nan")) is False
    assert t.save_if_best(float("inf")) is False
    assert t.best_val == float("-inf")
    assert t.best_model is None

    t.save_if_best(0.5)
    assert t.save_if_best(float("nan")) is False
    assert t.best_val == 0.5  # the NaN did not displace it


def test_only_improvements_are_kept():
    t = _stub(minimize=False)
    assert t.save_if_best(0.5) is True
    assert t.best_model is not None
    t.epoch = 1
    assert t.save_if_best(0.4) is False
    assert (t.best_val, t.best_epoch) == (0.5, 0)
    t.epoch = 2
    assert t.save_if_best(0.6) is True
    assert (t.best_val, t.best_epoch) == (0.6, 2)


def test_minimizing_keeps_the_lowest():
    t = _stub(minimize=True)
    assert t.save_if_best(0.5) is True
    assert t.save_if_best(0.6) is False
    assert t.save_if_best(0.4) is True
    assert t.best_val == 0.4


def test_patience_counts_down_and_refills():
    t = _stub(minimize=False, patience=2)
    for epoch, score, stop in [(0, 0.5, False), (1, 0.1, False), (2, 0.9, False), (3, 0.1, False)]:
        t.epoch = epoch
        assert t.step_patience(t.save_if_best(score)) is stop
    t.epoch = 4
    assert t.step_patience(t.save_if_best(0.1)) is True  # second miss since the refill


def test_the_countdown_waits_for_start_patience():
    t = _stub(minimize=False, patience=1, start_patience=3)
    for epoch in range(3):
        t.epoch = epoch
        assert t.step_patience(False) is False  # below start_patience, no countdown
    t.epoch = 3
    assert t.step_patience(False) is True


def test_unset_patience_lasts_the_whole_budget():
    """With no val.patience the counter falls back to num_epochs, so a run that never
    improves still uses its full budget rather than stopping partway."""
    t = _stub(minimize=False)  # num_epochs=10
    for epoch in range(9):
        t.epoch = epoch
        assert t.step_patience(False) is False
    t.epoch = 9
    assert t.step_patience(False) is True  # exhausted exactly at the budget


# ---------------------------------------------------------------- the attribute contract


def test_only_the_two_always_defined_names_exist_before_opting_in():
    obj = _Stub.__new__(_Stub)
    obj.cfg = OmegaConf.create({"num_epochs": 3, "val": {}})
    obj.best_model, obj.best_metrics = None, {}
    obj._best_tracking_initialized = False
    assert all(hasattr(obj, name) for name in ALWAYS_DEFINED)
    assert not any(hasattr(obj, name) for name in OPT_IN)


def test_tracking_attributes_are_never_probed_for():
    """``hasattr``/``getattr`` on these is the bug that made two trainers silently stop
    recording a best: the base defines them now, so a probe reads a seeded value rather
    than the absent-attribute default the caller wrote its comparison against."""
    offenders = []
    for path in sorted(SRC.rglob("*.py")):
        if path.name == "trainer.py" and path.parent.name == "core":
            continue
        source = path.read_text()
        for name in (*ALWAYS_DEFINED, *OPT_IN):
            for probe in (f'hasattr(self, "{name}")', f'getattr(self, "{name}"'):
                if probe in source:
                    offenders.append(f"{path.relative_to(SRC)}: {probe}")
    assert not offenders, "\n".join(offenders)
