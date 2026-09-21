"""Tests for resuming a run: what :meth:`BaseTrainer._restore_checkpoint_items` puts back.

The failure this guards against is quiet. A half-restored resume loads the weights, skips
the optimizer's gradient history and the scheduler's place on the LR curve, and trains on
from there, so the run reports nothing unusual while its learning rate has jumped back to
the start of the schedule. The check is that every object a trainer registers comes back,
and that a finetune load still gets a fresh optimizer.
"""

import re
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from omegaconf import OmegaConf

from core.trainer import BaseTrainer
from core.utils.checkpoint import save_ckpt

SRC = Path(__file__).parents[1]


class _Recorder:
    """Stands in for anything registered: remembers what was loaded into it."""

    def __init__(self, value):
        self.value = value
        self.loaded = None

    def state_dict(self):
        return {"value": self.value}

    def load_state_dict(self, state):
        self.loaded = state
        self.value = state["value"]


def _trainer(ckpt_cfg, to_ckpt):
    """A stand-in carrying only what the restore reads, so no run has to be built."""
    return SimpleNamespace(
        cfg=OmegaConf.create({"ckpt": ckpt_cfg}),
        to_ckpt=to_ckpt,
        logger=SimpleNamespace(info=lambda *a, **k: None, warn=lambda *a, **k: None),
    )


RESUME = {"resume": True, "model_only": False, "load_from": "last.pt"}


def test_a_resume_restores_every_registered_item():
    items = {"optimizer": _Recorder(1), "scheduler": _Recorder(2), "masker": _Recorder(3)}
    ckpt = {f"{name}_state_dict": {"value": 10 + i} for i, name in enumerate(items)}

    BaseTrainer._restore_checkpoint_items(_trainer(RESUME, items), ckpt)

    assert [it.value for it in items.values()] == [10, 11, 12]


def test_the_model_is_left_to_the_trainer():
    """The trainer's own load extends vocabularies and grafts task heads; a blind
    state_dict copy here would undo that."""
    items = {"model": _Recorder(1), "optimizer": _Recorder(2)}
    ckpt = {"model_state_dict": {"value": 99}, "optimizer_state_dict": {"value": 42}}

    BaseTrainer._restore_checkpoint_items(_trainer(RESUME, items), ckpt)

    assert items["model"].loaded is None
    assert items["optimizer"].value == 42


@pytest.mark.parametrize(
    "ckpt_cfg",
    [
        pytest.param({**RESUME, "resume": False}, id="finetune"),
        pytest.param({**RESUME, "model_only": True}, id="model_only"),
    ],
)
def test_a_finetune_load_keeps_a_fresh_optimizer(ckpt_cfg):
    items = {"optimizer": _Recorder(1)}

    BaseTrainer._restore_checkpoint_items(_trainer(ckpt_cfg, items), {"optimizer_state_dict": {}})

    assert items["optimizer"].loaded is None


def test_state_that_no_longer_fits_the_run_is_an_error_not_a_silent_skip():
    class Mismatched(_Recorder):
        def load_state_dict(self, state):
            raise ValueError("loaded state dict contains a parameter group that doesn't match")

    with pytest.raises(RuntimeError, match=re.escape("ckpt.model_only=true")):
        BaseTrainer._restore_checkpoint_items(
            _trainer(RESUME, {"optimizer": Mismatched(1)}), {"optimizer_state_dict": {}}
        )


def test_an_older_checkpoint_missing_the_state_still_loads():
    items = {"optimizer": _Recorder(1)}

    BaseTrainer._restore_checkpoint_items(_trainer(RESUME, items), {"model_state_dict": {}})

    assert items["optimizer"].loaded is None


def test_a_real_optimizer_and_scheduler_round_trip(tmp_path):
    """End to end through save_ckpt: the momentum and the LR both come back."""
    torch.manual_seed(0)
    layer = torch.nn.Linear(4, 2)
    opt = torch.optim.AdamW(layer.parameters(), lr=0.1)
    sched = torch.optim.lr_scheduler.StepLR(opt, step_size=1, gamma=0.5)
    for _ in range(3):
        layer(torch.randn(8, 4)).sum().backward()
        opt.step()
        sched.step()
        opt.zero_grad()
    save_ckpt(tmp_path / "last.pt", optimizer=opt, scheduler=sched)

    fresh_layer = torch.nn.Linear(4, 2)
    fresh_opt = torch.optim.AdamW(fresh_layer.parameters(), lr=0.1)
    fresh_sched = torch.optim.lr_scheduler.StepLR(fresh_opt, step_size=1, gamma=0.5)
    assert fresh_sched.get_last_lr() != sched.get_last_lr()

    ckpt = torch.load(tmp_path / "last.pt", map_location="cpu", weights_only=False)
    items = {"optimizer": fresh_opt, "scheduler": fresh_sched}
    BaseTrainer._restore_checkpoint_items(_trainer(RESUME, items), ckpt)

    assert fresh_sched.get_last_lr() == sched.get_last_lr()
    # exp_avg is the gradient history a half-restore throws away
    assert fresh_opt.state_dict()["state"][0]["exp_avg"].shape == torch.Size([2, 4])


def test_every_trainer_that_builds_an_optimizer_registers_it():
    """The restore is driven by the registry, so an optimizer held outside it is lost.

    ``self.optimizer = None`` does not count: a closed-form baseline has nothing to
    resume."""
    assigned = re.compile(r"self\.optimizer\s*=\s*([A-Za-z_][\w.]*)")
    offenders = []
    for path in [*SRC.rglob("*_trainer.py"), *SRC.rglob("*_pretrain.py")]:
        if path.parent.name == "tests":
            continue
        source = path.read_text()
        builds = any(rhs != "None" for rhs in assigned.findall(source))
        if builds and "add_checkpoint_items" not in source:
            offenders.append(str(path.relative_to(SRC)))
    assert not offenders, f"these build an optimizer but register nothing: {sorted(offenders)}"


def test_no_trainer_hand_rolls_what_the_base_class_restores():
    """One restore path, so gating cannot drift per trainer: the five that predated the
    base class treated ckpt.model_only differently from each other."""
    pattern = re.compile(r"(optimizer|scheduler|lr_scheduler)\.load_state_dict\(\s*ckpt")
    found = sorted(
        str(p.relative_to(SRC))
        for p in SRC.rglob("*.py")
        if p.parent.name != "tests" and pattern.search(p.read_text())
    )
    assert not found, f"restore by registering with add_checkpoint_items instead: {found}"
