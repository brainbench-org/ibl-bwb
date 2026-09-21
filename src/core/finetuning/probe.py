import torch.nn as nn
from omegaconf import DictConfig

from core.finetuning.base import FinetuningStrategy


class Probe(FinetuningStrategy):
    """Freeze all parameters except those matching unfrozen_prefixes.

    Args:
        unfrozen_prefixes: parameter name prefixes to keep trainable
    """

    def __init__(self, model: nn.Module, cfg: DictConfig, unfrozen_prefixes: list[str]):
        super().__init__(model, cfg)
        self.unfrozen_prefixes = unfrozen_prefixes

    def setup(self):
        for name, param in self.model.named_parameters():
            if not any(name.startswith(p) for p in self.unfrozen_prefixes):
                param.requires_grad = False

    def update(self, epoch: int): ...
