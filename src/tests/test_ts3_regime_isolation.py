"""Tests that ``pretrain/`` cannot read the eval regime, and that ``ts3`` can.

NuCLR and NEMO used to embed the eval units every val epoch and fit a zero-shot probe on
them. Nothing selected on that metric in code, but it sat in W&B next to the loss curves,
one epoch budget or lr grid away from being test-set selection. The fix is structural
rather than behavioural: no module under ``pretrain/`` names the eval regime at all, and
the one pass that encodes both regimes lives in ``ts3/extract.py``. A behavioural version
of this test passed while a stale monitor call site sat in the trainer, which is why the
invariant is now a name that cannot appear rather than an order of calls.

The second half guards the taxonomy: an extractor's package says which regime it can be
evaluated under, and that is a property of how the model represents a unit, not a config.
"""

import ast
import inspect
from importlib import import_module
from pathlib import Path

import pytest
import yaml

from ts3.models.base import Extractor
from ts3.models.transductive import TransductiveExtractor

SRC = Path(__file__).parents[1]


def _cfg(config: Path) -> dict:
    return yaml.safe_load(config.read_text()) or {}


# The models whose monitor probes pretrain units during training. Each owns both its
# pretrainer and its monitor: sharing one would put the eval regime back within reach.
MONITORED = ("nuclr", "nemo")
TRAINERS = tuple(f"pretrain/models/{m}/{m}_pretrain.py" for m in MONITORED)
MONITORS = tuple(f"pretrain/models/{m}/monitor.py" for m in MONITORED)

# All a trainer may hand its monitor: pretrain embeddings and their unit ids. Spelled
# out rather than read off the protocol, since the protocol is one of the things pinned
# to it. Anything else here would be a regime the trainer gets to choose.
MONITOR_PARAMS = ("train_embs", "train_uids")

EXTRACTOR_CONFIGS = sorted((SRC / "ts3" / "models").glob("*/*/configs/extractor/*.yaml"))
MONITOR_CONFIGS = [
    c
    for c in sorted((SRC / "pretrain").glob("models/*/configs/trainer/*.yaml"))
    if _cfg(c).get("monitor")
]


def _module(path: str) -> ast.Module:
    return ast.parse((SRC / path).read_text())


def _method(path: str, name: str) -> ast.FunctionDef:
    for node in ast.walk(_module(path)):
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return node
    raise AssertionError(f"{path} has no {name}")


def _pretrain_sources() -> list[Path]:
    return sorted((SRC / "pretrain").rglob("*.py"))


def _class(target: str):
    module_path, _, class_name = target.rpartition(".")
    cls = getattr(import_module(module_path), class_name, None)
    assert cls is not None, f"{target} is not importable"
    return cls


def test_pretrain_never_names_the_eval_regime():
    """The whole invariant, in one grep: ``regime`` is a parameter under ``pretrain/``.

    Nothing there chooses it, so encoding eval units is something only ``ts3`` can ask for.
    """
    offenders = [
        f"{path.relative_to(SRC)}:{node.lineno}"
        for path in _pretrain_sources()
        for node in ast.walk(ast.parse(path.read_text()))
        if isinstance(node, ast.Constant) and node.value == "eval"
    ]
    assert not offenders, "pretrain/ names the eval regime at: " + ", ".join(offenders)


def test_ts3_extract_names_the_eval_regime():
    """The other direction: the eval half of the embeddings file still gets written."""
    regimes = {
        node.value
        for node in ast.walk(_module("ts3/extract.py"))
        if isinstance(node, ast.Constant) and node.value in ("pretrain", "eval")
    }
    assert regimes == {"pretrain", "eval"}, f"ts3/extract.py encodes {regimes}"


def test_there_are_extractor_configs_to_check():
    assert EXTRACTOR_CONFIGS, "no extractor config group found under ts3/models/"


@pytest.mark.parametrize("config", EXTRACTOR_CONFIGS, ids=lambda p: p.stem)
def test_every_extractor_config_targets_an_extractor(config: Path):
    """A class rename, or a config pointing nowhere, fails here rather than mid-run."""
    assert issubclass(_class(_cfg(config)["_target_"]), Extractor)


@pytest.mark.parametrize("config", EXTRACTOR_CONFIGS, ids=lambda p: p.stem)
def test_every_extractor_config_sits_under_the_regime_it_implements(config: Path):
    """Reading per-session finetuning checkpoints is what transductive means.

    A model whose unit embedding is a free parameter has no row for a held-out unit, so it
    cannot be filed as inductive by editing a path; moving one of these is a claim about
    the model that has to be implemented, not declared.
    """
    regime = config.relative_to(SRC / "ts3" / "models").parts[0]
    assert regime in ("inductive", "transductive"), f"{config} is under {regime}/"
    cls = _class(_cfg(config)["_target_"])
    assert issubclass(cls, TransductiveExtractor) == (regime == "transductive")


@pytest.mark.parametrize("path", MONITORS, ids=MONITORED)
def test_every_online_monitor_reads_the_pretrain_regime_only(path: str):
    """Each model's monitor builds its own unit table, and only the pretrain half.

    The blanket scan above forbids the eval name anywhere under ``pretrain/``; this adds
    the positive half, that a monitor does name the regime it reads.
    """
    regimes = {
        node.value
        for node in ast.walk(_module(path))
        if isinstance(node, ast.Constant) and node.value in ("pretrain", "eval")
    }
    assert regimes == {"pretrain"}, f"{path} reads {regimes or 'no regime'}"


def test_the_monitor_configs_and_the_trainers_agree():
    """Neither list can grow without the other, so a new monitor cannot skip these tests."""
    assert {c.parents[2].name for c in MONITOR_CONFIGS} == set(MONITORED)


@pytest.mark.parametrize("config", MONITOR_CONFIGS, ids=lambda p: p.stem)
def test_every_configured_monitor_matches_the_protocol(config: Path):
    """The suite name now lives in YAML, where test_layering's Python scan cannot see it.

    That is the point, but it also means a typo or a signature drift here would surface
    only at the first val epoch of a real run.
    """
    cls = _class(_cfg(config)["monitor"]["_target_"])
    params = tuple(inspect.signature(cls.__call__).parameters)[1:]
    assert params == MONITOR_PARAMS, f"{config.name}: monitor takes {params}"


@pytest.mark.parametrize("path", TRAINERS, ids=MONITORED)
def test_every_monitor_call_site_passes_pretrain_units_only(path: str):
    """A call site the signature change missed cost a full extraction run to find."""
    calls = [
        node
        for node in ast.walk(_module(path))
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "monitor"
    ]
    assert calls, f"{path} never calls the monitor"
    for call in calls:
        passed = {kw.arg for kw in call.keywords}
        assert not call.args, f"{path}:{call.lineno} passes the monitor positional args"
        assert passed == set(MONITOR_PARAMS), f"{path}:{call.lineno} passes {sorted(passed)}"
