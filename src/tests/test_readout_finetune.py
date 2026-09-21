"""Tests that a single-task finetune starts from the pretrained task head.

The failure this guards against is silent and expensive. ``load_ckpt`` loads with
``strict=False``, so a readout key that does not line up leaves the head at its random
initialization, the run trains anyway, and the only trace is one line in the log. The
number that comes out is a finetune in name only.

The two models reach the same place by different routes. POSSM always builds a
:class:`MultitaskReadout`, so a one-task model produces the same
``readout.projections.<task>`` key the pretrained checkpoint holds and the load matches on
its own. POYO uses a bare ``nn.Linear``, whose ``readout.weight`` matches nothing in a
multitask checkpoint, so it grafts the head across by name.
"""

from types import SimpleNamespace

import torch

from ibl_bwb_eval.tasks import get_ts1_readout_spec
from pretrain.models.possm.possm import POSSM
from pretrain.models.poyo.poyo import POYO

UNITS = [f"u{i}" for i in range(8)]
TASK, OTHER = "choice", "wheel_speed"


def _dataset():
    return SimpleNamespace(
        context_window=1.0,
        get_unit_ids=lambda: UNITS,
        get_session_ids=lambda: ["s0"],
    )


def _possm(seed: int) -> POSSM:
    torch.manual_seed(seed)
    model = POSSM(dim=32, depth=0, rnn_dim=32, num_rnn_layers=1, dim_head=32)
    model.link_datasets(_dataset(), _dataset())
    return model


def _poyo(seed: int) -> POYO:
    torch.manual_seed(seed)
    model = POYO(latent_step=0.125, num_latents_per_step=4, dim=32, depth=1, dim_head=32)
    model.link_datasets(_dataset(), _dataset())
    return model


def test_possm_takes_the_pretrained_head_for_the_task_it_finetunes():
    pretrained = _possm(0)
    pretrained.configure_multitask_readout(
        {task: get_ts1_readout_spec(task) for task in (TASK, OTHER)}
    )
    ckpt = {"model_state_dict": pretrained.state_dict()}

    finetuned = _possm(99)  # a different init, so an unloaded head is visible
    finetuned.configure_readout(get_ts1_readout_spec(TASK))
    want = pretrained.readout.projections[TASK].weight.detach()
    assert not torch.equal(finetuned.readout.projections[TASK].weight.detach(), want)

    finetuned.load_ckpt(ckpt)

    assert torch.equal(finetuned.readout.projections[TASK].weight.detach(), want), (
        "POSSM finetune left its readout at init. The multitask projections are keyed by "
        "task name and a one-task model builds the same key, so this means the keying "
        "changed on one side."
    )


def _as_multitask_ckpt(model: POYO, task: str) -> dict:
    """A real POYO checkpoint with its readout renamed the way a multitask one stores it.

    The state dict has to be the real thing: ``InfiniteVocabEmbedding`` reads its vocab
    back out of it, so a dict holding only readout keys cannot be loaded at all.
    """
    state = {k: v for k, v in model.state_dict().items() if not k.startswith("readout.")}
    for part in ("weight", "bias"):
        state[f"readout.projections.{task}.{part}"] = model.state_dict()[f"readout.{part}"]
    return {"model_state_dict": state}


def test_poyo_grafts_the_head_out_of_a_multitask_checkpoint():
    """POYO's readout is a plain Linear, so nothing matches without the graft."""
    pretrained = _poyo(0)
    pretrained.configure_readout(get_ts1_readout_spec(TASK))
    ckpt = _as_multitask_ckpt(pretrained, TASK)
    want = pretrained.readout.weight.detach().clone()

    finetuned = _poyo(99)  # a different init, so an unloaded head is visible
    finetuned.configure_readout(get_ts1_readout_spec(TASK))
    assert not torch.equal(finetuned.readout.weight.detach(), want)

    finetuned.load_ckpt(ckpt)

    assert torch.equal(finetuned.readout.weight.detach(), want), (
        "POYO did not graft the task head out of the checkpoint"
    )


def test_poyo_says_so_when_the_checkpoint_has_no_head_for_the_task():
    """A checkpoint without this task's head is legitimate, so it warns rather than
    raising. It must not pass quietly: the alternative reading of a random head is a
    finetune that never finetuned."""
    pretrained = _poyo(0)
    pretrained.configure_readout(get_ts1_readout_spec(TASK))
    ckpt = _as_multitask_ckpt(pretrained, OTHER)  # a head, but for the wrong task

    finetuned = _poyo(99)
    finetuned.configure_readout(get_ts1_readout_spec(TASK))
    before = finetuned.readout.weight.detach().clone()

    # the model logs through the CLI logger, which does not reach caplog
    warnings: list[str] = []
    finetuned.logger = SimpleNamespace(
        info=lambda *a, **k: None, warning=warnings.append, warn=warnings.append
    )
    finetuned.load_ckpt(ckpt)

    assert torch.equal(finetuned.readout.weight.detach(), before)
    assert any(TASK in message for message in warnings), (
        f"no warning named the missing {TASK!r} head: {warnings}"
    )


def test_possm_keeps_a_fresh_head_for_a_task_the_pretrain_never_saw():
    """Finetuning on a task outside the pretrain set is allowed, and starts from init."""
    pretrained = _possm(0)
    pretrained.configure_multitask_readout({OTHER: get_ts1_readout_spec(OTHER)})

    finetuned = _possm(99)
    finetuned.configure_readout(get_ts1_readout_spec(TASK))
    before = finetuned.readout.projections[TASK].weight.detach().clone()

    finetuned.load_ckpt({"model_state_dict": pretrained.state_dict()})

    assert torch.equal(finetuned.readout.projections[TASK].weight.detach(), before)
