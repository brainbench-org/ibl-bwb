import sys
from collections.abc import Iterable
from pathlib import Path
from typing import Any, ClassVar

import git
import wandb
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm
from wandb.sdk.wandb_run import Run

from core.utils.util import get_cli_logger

EPOCH_PBAR_FMT = (
    "{l_bar}{bar}| Epoch {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}{postfix}]"
)
STEP_PBAR_FMT = (
    "{l_bar}{bar}| Step {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}{postfix}]"
)


def skip_if_dummy(func):
    def wrapper(self, *args, **kwargs):
        if self.dummy:
            return
        return func(self, *args, **kwargs)

    return wrapper


class Logger:
    r"""A wrapper for TQDM based progress bar and WandB."""

    _pbar = None
    _epoch_pbar = None
    log_to_pbar: ClassVar[dict] = {}
    log_to_wandb: ClassVar[dict] = {}
    dummy: bool = False
    wandb_run: Run | None = None
    run_id: str = None
    pbar_prefix: str = ""

    def __init__(self, rank: int, disable_pbar: bool = False):
        self.rank = rank
        self.dummy = rank != 0
        self.disable_pbar = disable_pbar

        self.cli_log = get_cli_logger(all_ranks=True)
        self.info(
            f"Rank {rank} online",
            all_ranks=True,
        )

    @skip_if_dummy
    def init_wandb(self, cfg: DictConfig, log_code: bool = False):
        if cfg.mode != "disabled":
            Path(cfg.dir).mkdir(exist_ok=True, parents=True)
        wandb.init(**cfg)  # type: ignore
        wandb.define_metric("train/step")
        wandb.define_metric("*", step_metric="train/step")
        assert wandb.run is not None
        self.wandb_run = wandb.run
        if log_code:
            # uploads a snapshot of the working tree, so it is opt-in
            self.wandb_run.log_code(root=".", include_fn=wandb_code_include_fn)
        self.run_id = self.wandb_run.id

    @skip_if_dummy
    def save_config(self, cfg: DictConfig):
        cfg = OmegaConf.to_container(cfg, resolve=True)

        try:
            repo = git.Repo(search_parent_directories=True)
            cfg["git"] = {"commit_id": repo.head.commit.hexsha}
            diff = repo.git.diff()
            if diff:
                cfg["git"]["diff"] = diff
        except Exception:
            # If git info can't be retrieved, continue without it
            pass

        wandb.config.update(cfg, allow_val_change=True)

    def info(self, msg, all_ranks=False):
        if all_ranks or not self.dummy:
            self.cli_log.info(msg, stacklevel=2)

    def warn(self, msg, all_ranks=False):
        if all_ranks or not self.dummy:
            self.cli_log.warn(msg, stacklevel=2)

    def error(self, msg):
        self.cli_log.error(msg, stacklevel=2)

    @skip_if_dummy
    def push(self):
        self._push_pbar()
        self._push_wandb()

    def _push_wandb(self):
        if self.wandb_run is None:
            return

        self.wandb_run.log(self.log_to_wandb)
        self.log_to_wandb = {}

    def _push_pbar(self):
        if self._pbar is None:
            return

        formatted_dict = {k: _pbar_value_format(v) for k, v in self.log_to_pbar.items()}
        desc = " | ".join([f"{k}: {v}" for k, v in formatted_dict.items()])
        if len(self.pbar_prefix) > 0:
            desc = f"[{self.pbar_prefix}] {desc}"
        desc += " "
        self._pbar.set_description(desc)
        self.log_to_pbar = {}

    @skip_if_dummy
    def log(self, name: str, value: Any, pbar: bool | str = False, wandb: bool = True):
        if pbar:
            if isinstance(pbar, str):
                self.log_to_pbar[pbar] = value
            else:
                self.log_to_pbar[name] = value

        if wandb:
            self.log_to_wandb[name] = value

    @skip_if_dummy
    def log_dict(self, data: dict[str, Any]):
        self.log_to_wandb.update(data)

    def get_pbar(self, iterable: Iterable, prefix: str = "", leave: bool = False):
        self._pbar = tqdm(
            iterable,
            desc=f"[{prefix}]",
            bar_format=STEP_PBAR_FMT,
            total=len(iterable),
            disable=self.dummy or self.disable_pbar,
            dynamic_ncols=True,
            leave=leave,
        )
        self.pbar_prefix = prefix

        yield from self._pbar

        self.pbar_prefix = ""

    def set_epoch_pbar(self, pbar):
        self._epoch_pbar = pbar

    @skip_if_dummy
    def update_epoch_postfix(self, postfix: dict):
        if self._epoch_pbar is not None:
            formatted = {k: _pbar_value_format(v) for k, v in postfix.items()}
            self._epoch_pbar.set_postfix(formatted)


def _pbar_value_format(value):
    if isinstance(value, str):
        return value

    if isinstance(value, (int, float)):
        if abs(value) < 0.001 or abs(value) > 1_000:
            return f"{value:.2e}"
        return f"{value:.3f}"

    return str(value)


# Mirrors .gitignore: data, checkpoints, run outputs and build products never belong
# in a code snapshot. Matched as whole path components, so a file whose own name
# contains one of these words is still uploaded.
CODE_SNAPSHOT_SKIP_DIRS = frozenset(
    {
        "__pycache__",
        ".venv",
        "build",
        "cache",
        "ckpt",
        "dist",
        "embs",
        "generated",
        "logs",
        "notebooks",
        "outputs",
        "pipeline_logs",
        "prds",
        "predictions",
        "processed",
        "raw",
        "results",
        "torch_brain",
        "venv",
        "wandb",
    }
)

CODE_SNAPSHOT_SUFFIXES = frozenset({".py", ".yaml", ".yml"})
CODE_SNAPSHOT_NAMES = frozenset({"pyproject.toml", "setup.sh"})


def wandb_code_include_fn(path: str, root: str):
    """Pick the files ``log_code`` uploads: sources and configs, no data or build output.

    Configs are in because a run is not reproducible from its code alone. wandb hands
    this absolute paths, so the skip list is matched against the path relative to
    ``root``.
    """
    _path = Path(path)
    if sys.prefix in str(_path):
        # Ignore any paths containing the Python path (virtual env)
        return False
    try:
        parents = Path(_path).relative_to(root).parts[:-1]
    except ValueError:
        return False
    if CODE_SNAPSHOT_SKIP_DIRS.intersection(parents):
        return False
    if any(part.endswith(".egg-info") for part in parents):
        return False
    return _path.suffix in CODE_SNAPSHOT_SUFFIXES or _path.name in CODE_SNAPSHOT_NAMES
