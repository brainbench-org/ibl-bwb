"""Shared entry point for every training script.

Each task suite keeps its own ``train.py`` so Hydra resolves ``config_path``
against that package's ``configs/``, but the launcher below is the same for all
of them.
"""

import os
import traceback
from collections.abc import Callable
from pathlib import Path

import hydra
import torch
import torch.multiprocessing as mp
import wandb
from dotenv import load_dotenv
from hydra.core.hydra_config import HydraConfig
from hydra.errors import ConfigCompositionException
from omegaconf import DictConfig, open_dict

from core.trainer import BaseTrainer
from core.utils.distributed import cleanup_ddp, get_open_port
from core.utils.logger import get_cli_logger

# Every entry point imports this, so .env lands in os.environ before Hydra resolves
# any ${oc.env:...}. Anchored to the checkout, not the CWD.
load_dotenv(Path(__file__).resolve().parents[2] / ".env")


def train(cfg: DictConfig, rank: int, world_size: int):
    """Instantiate ``cfg.trainer`` on one rank and run it to completion.

    Args:
        cfg: Fully composed Hydra config.
        rank: Local process rank (device index).
        world_size: Total number of processes in the distributed group.

    Returns:
        The trainer's final metrics dict, or ``None`` if it logged none.
    """
    if torch.cuda.is_available():
        torch.cuda.set_device(rank)  # avoid putting things on GPU0 by default

    try:
        trainer: BaseTrainer = hydra.utils.instantiate(
            cfg.trainer,
            cfg=cfg,
            rank=rank,
            world_size=world_size,
            _recursive_=False,
        )
        metrics = trainer.train()
        exit_code = 0
    except Exception as e:
        # ensures error is logged to wandb
        exit_code = 1
        wandb.summary["error"] = str(e)
        wandb.summary["traceback"] = traceback.format_exc()
        raise
    finally:
        # ensures cleanup even if training fails
        cleanup_ddp()
        wandb.finish(exit_code=exit_code)
    return metrics


def train_ddp(cfg: DictConfig):
    """Spawn one :func:`train` process per extra GPU and run rank 0 inline."""
    num_gpus = torch.cuda.device_count()
    world_size = max(num_gpus, 1)  # a CPU-only node is a single-process world

    if (num_gpus > 1 or cfg.ddp.force) and cfg.ddp.master_port is None:
        cfg.ddp.master_port = get_open_port()
        get_cli_logger().info(f"DDP MASTER_PORT = {cfg.ddp.master_port}")

    mp.set_start_method("spawn")
    for rank in range(1, num_gpus):
        mp.Process(target=train, args=(cfg, rank, world_size)).start()

    return train(cfg, 0, world_size)


def launch(cfg: DictConfig):
    """Record the Hydra choices on ``cfg``, then train. Call this from ``main``."""
    hydra_choices = HydraConfig.get().runtime.choices
    with open_dict(cfg):
        cfg.hydra_choices = hydra_choices

    return train_ddp(cfg)


def run(main: Callable[[], None]) -> None:
    """Call a Hydra entry point, reporting a missing config group the way a run does.

    Hydra composes the config to render ``--help``, but does that outside its own error
    handler, so ``train.py --help`` exits on a traceback where ``train.py`` prints one
    line. ``HYDRA_FULL_ERROR=1`` still gets the traceback, as everywhere else.
    """
    try:
        main()
    except ConfigCompositionException as e:
        if os.environ.get("HYDRA_FULL_ERROR"):
            raise
        raise SystemExit(
            f"{e}\nSet the environment variable HYDRA_FULL_ERROR=1 for a complete stack trace."
        ) from None
