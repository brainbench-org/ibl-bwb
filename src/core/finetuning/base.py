from abc import ABC, abstractmethod

import torch.nn as nn
from omegaconf import DictConfig

from core.utils.logger import get_cli_logger


class FinetuningStrategy(ABC):
    """Base class for finetuning strategies.

    Subclasses must implement:

    - setup: initialize frozen/unfrozen parameter state
    - update: called at the start of each epoch to adjust parameter state

    """

    def __init__(self, model: nn.Module, cfg: DictConfig):
        self.model = model
        self.cfg = cfg
        self.enable = cfg.finetuning.enable
        self.logger = get_cli_logger()

    @abstractmethod
    def setup(self): ...

    @abstractmethod
    def update(self, epoch: int): ...
