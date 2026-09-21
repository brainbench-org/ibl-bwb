"""Optimizer parameter grouping shared by the trainers and the TS3 probes."""

from collections.abc import Sequence

import torch

DEFAULT_NO_WEIGHT_DECAY = ("bias", "norm", "emb")


def param_groups(
    *modules: torch.nn.Module,
    weight_decay: float,
    no_weight_decay: Sequence[str] = DEFAULT_NO_WEIGHT_DECAY,
) -> list[dict]:
    """Split ``modules``' parameters into a weight-decayed and an undecayed optimizer group.

    A parameter skips weight decay if it is 1-D (biases, norm and other scale vectors) or
    if its name contains one of ``no_weight_decay``.
    """
    decayed, undecayed = [], []
    for module in modules:
        for name, param in module.named_parameters():
            group = (
                undecayed
                if param.ndim == 1 or any(nd in name for nd in no_weight_decay)
                else decayed
            )
            group.append(param)
    return [
        {"params": decayed, "weight_decay": weight_decay},
        {"params": undecayed, "weight_decay": 0.0},
    ]
