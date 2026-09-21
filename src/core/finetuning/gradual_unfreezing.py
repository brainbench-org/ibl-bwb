import torch.nn as nn
from omegaconf import DictConfig

from core.finetuning.base import FinetuningStrategy


class GradualUnfreezing(FinetuningStrategy):
    """Freeze all parameters except those matching unfrozen_prefixes, then unfreeze at a target epoch.

    Args:
        unfrozen_prefixes: parameter name prefixes to keep trainable during the frozen phase
        unfreeze_at_epoch: epoch at which the full model is unfrozen (0 disables freezing)
    """

    def __init__(
        self,
        model: nn.Module,
        cfg: DictConfig,
        unfreeze_at_epoch: int,
        unfrozen_prefixes: list[str],
    ):
        super().__init__(model, cfg)
        self.unfreeze_at_epoch = unfreeze_at_epoch
        self.unfrozen_prefixes = unfrozen_prefixes
        self.enable = self.enable and unfreeze_at_epoch != 0

    def setup(self):
        self.frozen_params = []
        for name, param in self.model.named_parameters():
            if not any(name.startswith(p) for p in self.unfrozen_prefixes) and param.requires_grad:
                param.requires_grad = False
                self.frozen_params.append(param)

        self.logger.info("Backbone frozen (performing calibration) starting at epoch 0.")

    def update(self, epoch: int):
        if self.enable and epoch == self.unfreeze_at_epoch:
            if self.frozen_params is None:
                raise RuntimeError("Model is not frozen, can't unfreeze")

            for param in self.frozen_params:
                param.requires_grad = True

            self.frozen_params = None

            self.logger.info(
                f"Backbone unfrozen (performing finetuning) starting at epoch {self.unfreeze_at_epoch}."
            )
