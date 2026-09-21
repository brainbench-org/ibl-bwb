import hashlib
import logging
import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist


class Precision:
    def __init__(self, precision_str: str):
        self.precision_str = precision_str

        if precision_str == "fp32":
            self.dtype = torch.float32
        elif precision_str == "fp16":
            self.dtype = torch.float16
        elif precision_str == "bf16":
            self.dtype = torch.bfloat16
        else:
            raise ValueError(f"Invalid precision string: {precision_str}")

    def __repr__(self):
        return self.precision_str

    def __str__(self):
        return self.precision_str


def get_device(rank: int) -> torch.device:
    return torch.device(rank) if torch.cuda.is_available() else torch.device("cpu")


def move_to_device(data, device: torch.device):
    if isinstance(data, torch.Tensor):
        return data.to(device)
    elif isinstance(data, dict):
        return {k: move_to_device(v, device) for k, v in data.items()}
    elif isinstance(data, list):
        return [move_to_device(v, device) for v in data]
    elif isinstance(data, tuple):
        return tuple(move_to_device(v, device) for v in data)
    elif isinstance(data, (str, int, float, bool, type(None), np.ndarray)):
        # Metadata/scalars do not need device transfer.
        return data
    else:
        raise TypeError(f"Unknown data type {type(data)}")


def seed_everything(seed):
    import random

    import numpy
    import torch

    random.seed(seed)
    numpy.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def rank_zero_only(func):
    def wrapper(self, *args, **kwargs):
        if self.rank != 0:
            return
        return func(self, *args, **kwargs)

    return wrapper


def expand_path(path: str | Path) -> Path:
    return Path(os.path.expandvars(os.path.expanduser(path)))


def _count_params(module: torch.nn.Module, trainable_only: bool = False):
    total = 0
    skipped = []
    for name, p in module.named_parameters():
        if trainable_only and not p.requires_grad:
            continue
        try:
            total += p.numel()
        except ValueError:
            skipped.append(name)
    return total, skipped


def get_num_params(module: torch.nn.Module, return_skipped: bool = True):
    total, skipped = _count_params(module, trainable_only=False)
    if return_skipped:
        return total, skipped
    return total


def get_num_trainable_params(module: torch.nn.Module, return_skipped: bool = True):
    total, skipped = _count_params(module, trainable_only=True)
    if return_skipped:
        return total, skipped
    return total


def log_param_breakdown(logger, model: torch.nn.Module, other_params: tuple[str, ...] = ()) -> None:
    """Log a model's parameter breakdown: total, backbone, trainable, other.

    ``other_params`` names the non-backbone submodules (stitchers, embedding tables);
    backbone is the remainder, and a name the model lacks counts as nothing.
    """
    total, skipped = get_num_params(model)
    trainable = get_num_trainable_params(model, return_skipped=False)
    other = sum(
        get_num_params(getattr(model, name), return_skipped=False)
        for name in other_params
        if hasattr(model, name)
    )
    backbone = total - other

    logger.info(f"Number of parameters: {human_readable(total)} ({total:,})")
    logger.info(f"Number of backbone parameters: {human_readable(backbone)} ({backbone:,})")
    logger.info(f"Number of trainable parameters: {human_readable(trainable)} ({trainable:,})")
    logger.info(f"Number of other parameters: {human_readable(other)} ({other:,})")
    if skipped:
        logger.info(f"Skipped lazy parameters: {skipped}")


def human_readable(n):
    if n >= 1_000_000_000:
        return f"{n / 1_000_000_000:.2f}B"
    elif n >= 1_000_000:
        return f"{n / 1_000_000:.2f}M"
    elif n >= 1_000:
        return f"{n / 1_000:.2f}K"
    return str(n)


class RelativePathFilter(logging.Filter):
    """Give every record a ``relpath``, which the CLI format prints instead of an abspath."""

    def filter(self, record):
        if hasattr(record, "pathname"):
            record.relpath = os.path.relpath(record.pathname)
        return True


LOGGING_FORMAT = (
    "\033[1;33m%(levelname)s\033[0m | "
    "\033[1;36m%(asctime)s\033[0m | "
    "\033[1;32m%(relpath)s:%(lineno)-4d\033[0m | "
    "%(message)s"
)


def get_cli_logger(all_ranks=False):
    logger = logging.getLogger()

    # Not basicConfig(force=True): force drops every handler, Hydra's run-log writer
    # included. FileHandler subclasses StreamHandler, hence the second isinstance.
    for handler in list(logger.handlers):
        if isinstance(handler, logging.StreamHandler) and not isinstance(
            handler, logging.FileHandler
        ):
            logger.removeHandler(handler)
    console = logging.StreamHandler(sys.stdout)
    console.setFormatter(logging.Formatter(LOGGING_FORMAT))
    logger.addHandler(console)
    logger.setLevel(logging.INFO)

    # On the handler, not the logger: logger-level filters do not run for records
    # propagating up from module loggers, which then hit %(relpath)s without one. Handlers
    # now outlive the call, so only filter one that is not already filtered.
    for handler in logger.handlers:
        if not any(isinstance(f, RelativePathFilter) for f in handler.filters):
            handler.addFilter(RelativePathFilter())
    if not all_ranks and dist.is_initialized() and dist.get_rank() != 0:
        logger.setLevel(logging.ERROR)
    return logger


def is_divisible(x: float, y: float, tol: float = 1e-6) -> bool:
    """Returns true if x is divisible by y."""
    return abs(x / y - round(x / y)) < tol


def kfold_assignment(ids: list[str], folds: int) -> list[int]:
    """Assign each id to a fold by hashing it, so the split is stable across runs."""

    def deterministic_hash(x: str) -> int:
        return int(hashlib.sha256(x.encode("utf-8")).hexdigest(), 16)

    return [deterministic_hash(x) % folds for x in ids]


def global_mean_pool(x: torch.Tensor, batch: torch.Tensor, n: int) -> torch.Tensor:
    """Mean of ``x`` within each of the ``n`` groups named by ``batch``, zero where empty."""
    out = torch.zeros(n, dtype=x.dtype, device=x.device)
    out.scatter_add_(0, batch, x)
    counts = torch.bincount(batch, minlength=n).float()
    return out / counts.clamp(min=1)
