"""Tests for per-parameter weight and gradient logging.

This already rotted once: ``log_weights`` and ``log_grads`` sat in every train.yaml
while nothing called the methods reading them. The gate has to stay reachable from
every trainer that steps, read before the clip, and stay off and throttled by default.
"""

from pathlib import Path

import pytest
import torch
from omegaconf import OmegaConf

from core.trainer import BaseTrainer

SRC = Path(__file__).parents[1]
PACKAGES = ("ts1", "ts2", "ts3", "pretrain")

# Plain functions: no trainer, no logger to log into. The probe clips on its own.
EXEMPT = ("ts3/probes/mlp.py", "core/utils/check_xformers_fa.py")


class _Stub:
    """Only the trainer attributes the two logging methods touch."""

    clip_and_log_grad_norm = BaseTrainer.clip_and_log_grad_norm
    log_param_grad_stats = BaseTrainer.log_param_grad_stats

    def __init__(self, model: torch.nn.Module, **cfg):
        self.cfg = OmegaConf.create({"log_weights": False, "log_grads": False, **cfg})
        self.model = model
        self.train_step = 0
        self.device = torch.device("cpu")
        self._grad_norms = []
        self.logged = {}
        self.logger = type(
            "L",
            (),
            {
                "log_dict": lambda _s, d: self.logged.update(d),
                "log": lambda _s, k, v, **kw: self.logged.__setitem__(k, v),
            },
        )()


def _stub(**cfg) -> _Stub:
    """Model with one frozen parameter and one left without a grad."""
    torch.manual_seed(0)
    model = torch.nn.Sequential(torch.nn.Linear(4, 3), torch.nn.LayerNorm(3))
    model[0].bias.requires_grad_(False)
    trainer = _Stub(model, **cfg)
    model(torch.randn(8, 4)).pow(2).sum().backward()
    model[1].weight.grad = None
    return trainer


# ------------------------------------------------------------------------ the gate works


def test_off_by_default():
    trainer = _stub()
    trainer.log_param_grad_stats()
    assert trainer.logged == {}, "logging happens with both gates false"


@pytest.mark.parametrize(
    ("gate", "prefixes"),
    [
        ("log_grads", ("grad_norm/", "grad_weight_ratio/")),
        ("log_weights", ("weight_norm/",)),
    ],
)
def test_gate_produces_its_keys(gate, prefixes):
    """The rot mode: a gate reads true and still logs nothing."""
    trainer = _stub(**{gate: True}, log_stats_every_n_steps=1)
    trainer.log_param_grad_stats()
    assert trainer.logged, f"{gate}=true logged nothing"
    for prefix in prefixes:
        assert any(k.startswith(prefix) for k in trainer.logged), f"{gate}=true logged no {prefix}*"
    other = "weight_norm/" if gate == "log_grads" else "grad_norm/"
    assert not any(k.startswith(other) for k in trainer.logged), f"{gate} leaked {other}* keys"


def test_frozen_and_gradless_params():
    trainer = _stub(log_grads=True, log_weights=True, log_stats_every_n_steps=1)
    trainer.log_param_grad_stats()
    assert "weight_norm/0.bias" not in trainer.logged, "frozen parameter reported"
    assert "grad_norm/1.weight" not in trainer.logged, "parameter without a grad reported a grad"
    assert "weight_norm/1.weight" in trainer.logged, "gradless parameter lost its weight norm"


def test_ratio_matches_the_two_norms_and_skips_zero_weights():
    trainer = _stub(log_grads=True, log_weights=True, log_stats_every_n_steps=1)
    trainer.log_param_grad_stats()
    w = trainer.logged["weight_norm/0.weight"]
    g = trainer.logged["grad_norm/0.weight"]
    assert trainer.logged["grad_weight_ratio/0.weight"] == pytest.approx(g / w)
    # zero-init bias: ratio undefined, so absent rather than NaN
    assert trainer.logged["weight_norm/1.bias"] == 0.0
    assert "grad_weight_ratio/1.bias" not in trainer.logged


def test_ddp_and_single_gpu_share_key_names():
    """DDP prefixes parameters with ``module.``; keys must not fork on it."""
    torch.manual_seed(0)
    inner = torch.nn.Linear(4, 3)
    inner(torch.randn(8, 4)).pow(2).sum().backward()

    class _Wrapped(torch.nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module

    plain = _Stub(inner, log_grads=True, log_stats_every_n_steps=1)
    plain.log_param_grad_stats()
    wrapped = _Stub(_Wrapped(inner), log_grads=True, log_stats_every_n_steps=1)
    wrapped.log_param_grad_stats()
    assert plain.logged.keys() == wrapped.logged.keys()


def test_extra_modules_are_included():
    trainer = _stub(log_weights=True, log_stats_every_n_steps=1)
    head = torch.nn.Linear(3, 2)
    trainer.log_param_grad_stats(trainer.model, head)
    assert "weight_norm/weight" in trainer.logged, "parameters passed in explicitly were dropped"


# -------------------------------------------------------------------------- the throttle


@pytest.mark.parametrize(
    ("step", "every_n", "expected"),
    [
        (0, 100, True),
        (37, 100, False),
        (200, 100, True),
        (1, 1, True),
        (5, 0, False),
    ],
)
def test_throttle(step, every_n, expected):
    trainer = _stub(log_grads=True, log_stats_every_n_steps=every_n)
    trainer.train_step = step
    trainer.log_param_grad_stats()
    assert bool(trainer.logged) is expected


@pytest.mark.parametrize("pkg", PACKAGES)
def test_missing_knob_throttles_like_the_shipped_default(pkg):
    """The code fallback and the shipped value must not drift apart."""
    shipped = OmegaConf.load(SRC / pkg / "configs" / "train.yaml").log_stats_every_n_steps
    absent = _stub(log_grads=True)
    present = _stub(log_grads=True, log_stats_every_n_steps=shipped)
    for trainer in (absent, present):
        trainer.train_step = 1
        trainer.log_param_grad_stats()
    assert absent.logged.keys() == present.logged.keys(), (
        f"{pkg} ships log_stats_every_n_steps={shipped}, but the code falls back to "
        "something else, so a config without the key throttles differently"
    )


# ----------------------------------------------------------------------- pre-clip, wired


def test_stats_are_read_before_the_clip():
    trainer = _stub(log_grads=True, log_stats_every_n_steps=1, grad_clip=1e-6)
    pre = trainer.model[0].weight.grad.norm().item()
    trainer.clip_and_log_grad_norm()
    assert trainer.model[0].weight.grad.norm().item() < pre, "clip did not fire; test is vacuous"
    logged = trainer.logged["grad_norm/0.weight"]
    assert logged == pytest.approx(pre, rel=1e-5), "read after the clip: only ever the threshold"
    assert "train/grad_norm" in trainer.logged, "the existing global norm stopped being logged"


def test_the_clip_spans_every_module_it_is_given():
    """NuCLR steps an objective beside the model, and a norm over the model alone would
    leave that objective's gradients unclipped."""
    trainer = _stub(grad_clip=1e-6)
    objective = torch.nn.Linear(4, 3)
    objective(torch.randn(8, 4)).pow(2).sum().backward()
    pre = objective.weight.grad.norm().item()
    trainer.clip_and_log_grad_norm(trainer.model, objective)
    assert objective.weight.grad.norm().item() < pre, "a module passed in went unclipped"


@pytest.mark.parametrize("pkg", PACKAGES)
def test_every_suite_config_carries_the_keys(pkg):
    cfg = OmegaConf.load(SRC / pkg / "configs" / "train.yaml")
    assert cfg.log_weights is False and cfg.log_grads is False, f"{pkg} ships the gates on"
    assert cfg.log_stats_every_n_steps > 0, f"{pkg} has no usable throttle"


@pytest.mark.parametrize("pkg", PACKAGES)
def test_every_suite_root_states_the_clip(pkg):
    """An absent grad_clip reads as no clip, so a root that omits it hides the policy."""
    cfg = OmegaConf.load(SRC / pkg / "configs" / "train.yaml")
    assert "grad_clip" in cfg, f"{pkg} leaves grad_clip unset, which silently means no clip"


def _stepping_trainers() -> list[str]:
    """Every trainer source that runs backward(), minus the ones with nothing to log into."""
    return sorted(
        rel
        for path in SRC.rglob("*.py")
        if (rel := path.relative_to(SRC).as_posix()) not in EXEMPT
        and path.parent.name != "tests"
        and ".backward()" in (source := path.read_text())
        and "def clip_and_log_grad_norm" not in source
    )


def test_every_optimizer_step_reaches_the_hook():
    """A trainer that steps without reaching the gate logs nothing, which is the rot."""
    unreachable = [
        rel
        for rel in _stepping_trainers()
        if "clip_and_log_grad_norm(" not in (SRC / rel).read_text()
    ]
    assert not unreachable, (
        f"these call backward() but can never log param/grad stats: {unreachable}. "
        "Call self.clip_and_log_grad_norm() after backward(), or add it to EXEMPT."
    )


def test_the_exempt_probe_still_clips(monkeypatch):
    """EXEMPT buys ts3/probes/mlp.py out of the logging hook, not out of the clip."""
    from ts3.probes import mlp

    hparams = {
        "m.hidden_layers": 1,
        "m.hidden_dim_log2": 3,
        "m.dropout": 0.0,
        "m.batch_norm": False,
        "m.activation": "relu",
        "num_epochs": 1,
        "batch_size_log2": 2,  # 8 samples, so two steps
        "lr": 1e-3,
        "wd": 0.0,
    }
    x, y = torch.randn(8, 4), torch.tensor([0, 1] * 4)[:, None]

    seen = []
    monkeypatch.setattr(
        torch.nn.utils, "clip_grad_norm_", lambda params, max_norm: seen.append(max_norm)
    )
    for grad_clip, expected in ((0.7, [0.7, 0.7]), (None, [])):
        seen.clear()
        mlp.train_one_mlp(x, y, hparams, torch.device("cpu"), 2, grad_clip, seed=0)
        assert seen == expected, f"grad_clip={grad_clip} clipped {seen}, expected {expected}"


@pytest.mark.parametrize("rel", _stepping_trainers())
def test_the_hook_runs_after_backward_and_before_the_step(rel):
    """Before backward() it reports the previous step's gradients; after the step it
    clips weights that have already moved."""
    lines = (SRC / rel).read_text().splitlines()
    backward = next(i for i, line in enumerate(lines) if ".backward()" in line)
    call = next(i for i, line in enumerate(lines) if "self.clip_and_log_grad_norm(" in line)
    assert call > backward, f"{rel} clips and logs before backward()"
    step = next(
        i for i, line in enumerate(lines[backward:], backward) if ".optimizer.step()" in line
    )
    assert call < step, f"{rel} clips and logs after the weights have been updated"
