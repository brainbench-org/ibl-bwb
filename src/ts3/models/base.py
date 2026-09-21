"""What a TS3 extractor is: a checkpoint in, one embedding per unit out.

The regime is an argument, never a field. ``extract.py`` asks one extractor object for the
pretrain units and then the eval units, and the two halves of the file it writes are only
comparable because the same object answered both calls. Which checkpoint answers the eval
call is the entire inductive/transductive difference, and it is settled inside
:meth:`Extractor.encode`, not by the caller.
"""

from abc import ABC, abstractmethod
from pathlib import Path

import hydra
import numpy as np
import torch
from omegaconf import DictConfig

from core.dataset import BenchmarkRegime
from core.utils.logger import Logger


class Extractor(ABC):
    """A unit-embedding producer, driven by ``ts3/extract.py``."""

    def setup(self, data_root: Path, device: torch.device, logger: Logger) -> None:
        """Bind the run's context. Override to load checkpoints once, not per regime."""
        self.data_root = data_root
        self.device = device
        self.logger = logger

    @property
    @abstractmethod
    def name(self) -> str:
        """Names the embeddings file, so two runs of one model do not overwrite each other."""

    @property
    def run_seed(self) -> int | None:
        """The seed of the run behind these embeddings, or None when no run produced them."""
        return None

    @abstractmethod
    def encode(self, regime: BenchmarkRegime) -> tuple[torch.Tensor, np.ndarray]:
        """Embeddings and uids for every unit of ``regime`` this extractor can represent.

        Returns (N, D) and (N,) in matching order. Units the extractor cannot represent may
        be omitted: the probe join drops anything TS3 does not score, and insists
        separately that the eval side is complete.
        """


def load_pretrained(
    ckpt_path: Path, device: torch.device, logger: Logger, **overrides
) -> tuple[torch.nn.Module, DictConfig]:
    """Rebuild a checkpoint's model from the config it was trained with.

    ``overrides`` replaces entries of that config when not None: a forward pass fits more
    per batch than a backward one, and the machine encoding need not be the one that
    trained. They land after the model is built, so they reach what the caller reads off
    the returned config (``batch_size``, ``num_workers``) and never the model itself.
    """
    ckpt = torch.load(ckpt_path, weights_only=False, map_location="cpu")
    cfg = DictConfig(ckpt["cfg"])
    logger.info(f"Checkpoint: {ckpt_path} (epoch {ckpt.get('epoch')})")
    logger.info(f"Model: {cfg.model._target_}")

    model = hydra.utils.instantiate(cfg.model)
    model.load_state_dict(ckpt["model_state_dict"])

    for key, value in overrides.items():
        if value is not None:
            logger.info(f"{key}: {value} (the run used {cfg[key]})")
            cfg[key] = value

    return model.to(device).eval(), cfg
