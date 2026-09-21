"""The training loop every model's trainer is built on."""

import math
import subprocess
from abc import abstractmethod
from copy import deepcopy
from itertools import chain, islice

import torch
import torch.distributed as dist
from omegaconf import DictConfig, OmegaConf
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from tqdm import tqdm

from core.optim import DEFAULT_NO_WEIGHT_DECAY, param_groups
from core.utils.checkpoint import save_ckpt
from core.utils.distributed import cleanup_ddp, setup_ddp
from core.utils.logger import EPOCH_PBAR_FMT, Logger
from core.utils.util import (
    Precision,
    expand_path,
    get_device,
    rank_zero_only,
    seed_everything,
)

__api_ref__ = {
    "description": None,
    "sections": [{"title": None, "autosummary": ["BaseTrainer"]}],
}


class _OneBatchLoader:
    """A loader capped at its first batch, for ``cfg.debug`` runs.

    Iteration and ``len`` see the cap, everything else passes through to the loader.
    """

    def __init__(self, loader: DataLoader):
        self.loader = loader

    def __iter__(self):
        return islice(iter(self.loader), 1)

    def __len__(self):
        return min(len(self.loader), 1)  # an empty split stays empty

    def __getattr__(self, name):
        return getattr(self.loader, name)


class BaseTrainer:
    """The epoch loop, and everything around it that depends on neither task nor model.

    Logging, checkpointing, DDP, early stopping and best-model selection live here;
    a subclass supplies :meth:`setup` and the epoch internals.
    """

    def __init__(self, cfg: DictConfig, rank: int, world_size: int):
        """Initialize the base trainer.

        Sets up distributed training, logging, and loads a checkpoint if specified.
        Calls :meth:`setup` to initialize the model, data loaders, and optimizers.

        Args:
            cfg: Hydra configuration object.
            rank: Local process rank (device index).
            world_size: Total number of processes in the distributed group.
        """
        assert cfg.get("data_root"), (
            "data_root is unset: set BWB_DATA_ROOT, its per-build override, or pass data_root="
        )
        seed_everything(cfg.seed)
        setup_ddp(rank, world_size, cfg.ddp)

        self.cfg = cfg
        self.precision = Precision(cfg.precision)
        self.rank = rank
        self.world_size = world_size
        self.is_distributed = dist.is_initialized()
        self.device = get_device(rank)
        self.train_step = 0
        self.epoch = 0
        self._grad_norms = []
        self.to_ckpt = {}

        if self.cfg.debug:
            self.cfg.wandb.mode = "disabled"
            self.cfg.ckpt.enable = False
            self.cfg.num_workers = 0
            self.cfg.num_epochs = min(self.cfg.num_epochs, 1)  # 0 stays eval-only
            if "grad_accum_steps" in self.cfg:
                self.cfg.grad_accum_steps = 1  # else no batch ever reaches an optimizer step
            if OmegaConf.select(self.cfg, "save_preds.enable"):
                self.cfg.save_preds.enable = False  # a truncated file scores as a whole one

        self._setup_logger()
        self.logger.save_config(self.cfg)

        self.best_model = None
        self.best_metrics = {}
        self._best_tracking_initialized = False

        ckpt = self._load_ckpt()
        self.setup(ckpt)
        self._restore_checkpoint_items(ckpt)
        if self.cfg.debug:
            self._truncate_loaders()
        self._setup_ckpt()

    def __del__(self):
        cleanup_ddp()

    def train(self) -> dict | None:
        """Run the full training loop.

        Iterates over epochs, calling :meth:`train_epoch` each epoch and
        :meth:`val_epoch` every ``val.every_n_epochs`` epochs and at the final epoch.
        Checkpoints are saved after each epoch. Supports early stopping via the
        return value of :meth:`val_epoch`.

        Returns:
            Final metrics dict, or ``None`` if no metrics were logged.
        """
        if self.cfg.val.first:
            self.val_epoch()

        epoch_pbar = tqdm(
            range(self.epoch, self.cfg.num_epochs),
            initial=self.epoch,
            total=self.cfg.num_epochs,
            bar_format=EPOCH_PBAR_FMT,
            dynamic_ncols=True,
            disable=(self.rank != 0) or self.cfg.get("disable_pbar", False),
        )
        self.logger.set_epoch_pbar(epoch_pbar)

        early_stop = False
        for epoch in epoch_pbar:
            self.epoch = epoch
            self.barrier()
            self.train_epoch()
            self.log_epoch_grad_norm()
            self._save_checkpoint()
            self._save_last_checkpoint()

            should_do_val = (self.epoch % self.cfg.val.every_n_epochs == 0) or (  # if user asked
                self.epoch == self.cfg.num_epochs - 1  # if last epoch
            )
            if should_do_val:
                early_stop = self.val_epoch()

            if early_stop:
                break

        if self.cfg.test.get("enable", True):
            self.test()

        self.log_training_results()

        return self.__dict__.get("return_value", None)

    @abstractmethod
    def train_epoch(self):
        """Perform a training epoch."""
        ...

    @abstractmethod
    def val_epoch(self):
        """Perform a validation epoch.

        Called every `val.every_n_epochs` epochs, and at the end of training.

        Returns:
            early_stop : bool, True if early stopping should be triggered, False (or falsey) otherwise.
        """
        ...

    def test(self):
        """Perform a test epoch.

        Called at the end of training if `test.enable` is True.
        """
        self.logger.warn("Test method not implemented for this Trainer, doing nothing.")

    def log_training_results(self):
        """Log the training results, at the end of training."""
        ...

    @abstractmethod
    def setup(self, ckpt: dict | None):
        """Set up the trainer.

        This method is called after setting up distributed, loading the checkpoint, and setting up logging.

        Args:
            ckpt: The checkpoint dictionary, if any.
        """
        ...

    def _setup_logger(self):
        self.logger = Logger(self.rank, disable_pbar=self.cfg.get("disable_pbar", False))
        self.logger.init_wandb(self.cfg.wandb, log_code=self.cfg.get("log_code", False))
        self.logger.info(f"Run ID: {self.logger.run_id} (seed {self.cfg.seed})")
        if self.cfg.wandb.mode != "online":
            self.logger.warn(f"W&B mode: {self.cfg.wandb.mode}")

    def _truncate_loaders(self):
        """Cap every split at one batch.

        Runs after :meth:`setup`, so whatever was sized from ``len(train_loader)`` there,
        a scheduler above all, is unaffected.
        """
        for name in ("train_loader", "val_loader", "test_loader", "eval_loader"):
            loader = getattr(self, name, None)
            if loader is not None:
                setattr(self, name, _OneBatchLoader(loader))
        self.logger.warn(
            f"Debug mode: one batch per split, {self.cfg.num_epochs} epoch(s), "
            "no W&B or checkpoints"
        )

    def add_checkpoint_items(self, **kwargs):
        """Add items to the checkpoint dictionary, saved and restored by state_dict.

        Registering is what makes an object survive a resume, so every rank does it;
        writing the file stays rank zero's job.

        Args:
            kwargs: The items to add to the checkpoint dictionary.
        """
        self.to_ckpt.update(kwargs)

    def _restore_checkpoint_items(self, ckpt: dict | None):
        """Load the state of everything :meth:`add_checkpoint_items` registered.

        Runs once :meth:`setup` has returned and the registry is complete, so
        registering an optimizer is all a trainer needs for a resume that continues
        rather than restarts. Only under ``ckpt.resume``, since a finetune wants the
        weights and a fresh optimizer. The model is excluded: loading it belongs to the
        trainer, which may have to extend a vocabulary or graft a task head.
        """
        if ckpt is None or not self.cfg.ckpt.resume or self.cfg.ckpt.model_only:
            return

        restored, absent = [], []
        for name, item in self.to_ckpt.items():
            if name == "model" or not hasattr(item, "load_state_dict"):
                continue
            key = f"{name}_state_dict"
            if key not in ckpt:
                absent.append(key)
                continue
            target = item.module if isinstance(item, DDP) else item
            try:
                target.load_state_dict(ckpt[key])
            except (ValueError, RuntimeError) as e:
                raise RuntimeError(
                    f"Cannot restore '{name}' from {self.cfg.ckpt.load_from}: {e}. State "
                    "that no longer fits the run means this is a finetune rather than a "
                    "resume: pass ckpt.model_only=true or ckpt.resume=false."
                ) from None
            restored.append(name)

        if restored:
            self.logger.info(f"Resumed state: {', '.join(restored)}")
        if absent:
            self.logger.warn(f"Checkpoint carries no {', '.join(absent)}; left at init")

    @rank_zero_only
    def _setup_ckpt(self):
        if not self.cfg.ckpt.enable:
            self.logger.warn("Checkpointing disabled")
            self.ckpt_dir = None
            return

        # Create checkpoint directory, e.g. <ckpt.dir>/<trainer>_<run_dir>/best.pt,
        # loaded back with ckpt.load_from=<trainer>_<run_dir>/best.pt.
        trainer = (self.cfg.get("hydra_choices") or {}).get("trainer")
        run_dir = f"{trainer}_{self.logger.run_id}" if trainer else self.logger.run_id
        self.ckpt_dir = (expand_path(self.cfg.ckpt.dir) / run_dir).resolve()
        self.ckpt_dir.mkdir(exist_ok=True, parents=True)
        self.logger.info(f"Checkpoint directory: {self.ckpt_dir}")

        # Save git diff patch and commit ID (best effort: absent outside a git checkout)
        try:
            diff_path = self.ckpt_dir / "diff.patch"
            diff_path.write_bytes(subprocess.check_output(["git", "diff", "HEAD"]))
            commit_id = subprocess.check_output(["git", "rev-parse", "HEAD"]).decode("utf-8")
            (self.ckpt_dir / "commit_id.txt").write_text(commit_id.strip())
            self.logger.wandb_run.log_artifact(diff_path)
            self.logger.info("Saved diff.patch and commit_id.txt")
        except Exception as e:
            self.logger.warn(f"Could not record git state: {e}")

        # Tell user what will be checkpointed
        if len(self.to_ckpt) > 0:
            self.logger.info(f"Checkpoints will track: {list(self.to_ckpt.keys())}")
        else:
            self.logger.warn("No items specified for checkpointing")

    @rank_zero_only
    def _save_checkpoint(self):
        if self.ckpt_dir is None:
            return

        every_n_epochs = self.cfg.ckpt.every_n_epochs
        if (every_n_epochs is None) or (self.epoch % every_n_epochs != 0):
            return

        self.last_ckpt_path = save_ckpt(
            filepath=self.ckpt_dir / f"epoch_{self.epoch}.pt",
            train_step=self.train_step,
            epoch=self.epoch,
            run_id=self.logger.run_id,
            cfg=self.cfg,
            **self.to_ckpt,
        )

    @rank_zero_only
    def _save_last_checkpoint(self):
        if self.ckpt_dir is None:
            return
        if not self.cfg.ckpt.get("save_last", True):
            return

        self.last_ckpt_path = save_ckpt(
            filepath=self.ckpt_dir / "last.pt",
            train_step=self.train_step,
            epoch=self.epoch,
            run_id=self.logger.run_id,
            cfg=self.cfg,
            **self.to_ckpt,
        )

    @rank_zero_only
    def _save_best_checkpoint(self):
        if self.ckpt_dir is None:
            return

        self.best_ckpt_path = save_ckpt(
            filepath=self.ckpt_dir / "best.pt",
            train_step=self.train_step,
            epoch=self.epoch,
            run_id=self.logger.run_id,
            cfg=self.cfg,
            **self.to_ckpt,
        )

    def _load_ckpt(self) -> dict | None:
        if self.cfg.ckpt.load_from is None:
            return

        ckpt_path = expand_path(self.cfg.ckpt.load_from)
        if not ckpt_path.exists():
            # try this other path
            ckpt_path = expand_path(self.cfg.ckpt.dir) / self.cfg.ckpt.load_from
            if not ckpt_path.exists():
                raise ValueError(
                    f"Checkpoint not found at {self.cfg.ckpt.load_from} or {ckpt_path}"
                )
        self.cfg.ckpt.load_from = ckpt_path

        self.logger.info(f"Loading checkpoint {ckpt_path}")
        ckpt = torch.load(
            ckpt_path,
            map_location=self.device,
            weights_only=False,
        )

        if self.cfg.ckpt.resume:
            self.epoch = ckpt["epoch"] + 1
            self.train_step = ckpt["train_step"]
            self.logger.info(f"Resuming from: epoch = {self.epoch}, train_step = {self.train_step}")

        return ckpt

    def make_ddp(self, module: torch.nn.Module, find_unused_parameters: bool = False):
        """Wrap a module in DistributedDataParallel if running in distributed mode.

        Args:
            module: The module to wrap.
            find_unused_parameters: Passed to DDP. Set to ``True`` if some parameters
                are not used in the forward pass. Defaults to ``False``.

        Returns:
            torch.nn.Module: The DDP-wrapped module, or the original module if not distributed.
        """
        if self.is_distributed:
            ret = torch.nn.SyncBatchNorm.convert_sync_batchnorm(module.to(self.rank))
            ret = DDP(
                ret,
                device_ids=[self.rank],
                find_unused_parameters=find_unused_parameters,
            )
            return ret
        else:
            return module

    def barrier(self):
        """Synchronize all processes in the distributed group."""
        if self.is_distributed:
            dist.barrier()

    def reduce_mean(self, total: float, count: float) -> float:
        """Mean over the whole split rather than over this rank's shard of it.

        A distributed sampler hands every rank a different slice, so a locally divided
        sum leaves the ranks disagreeing on the score they select on. Every rank must
        call this the same number of times, in the same order.

        Args:
            total: This rank's summed quantity.
            count: This rank's number of terms in that sum.

        Returns:
            The pooled mean, NaN if no rank contributed a term.
        """
        if self.is_distributed:
            pair = torch.tensor([total, count], dtype=torch.float64, device=self.device)
            dist.all_reduce(pair)
            total, count = pair.tolist()
        return float("nan") if count == 0 else total / count

    def reset_best_tracking(self, minimize: bool | None = None):
        """Initialize the best-score bookkeeping used by :meth:`save_if_best`.

        :meth:`save_if_best` calls this itself on first use, so a trainer only needs it
        to pin a direction that config does not carry: pass ``minimize`` when the
        trainer fixes it in code, or call it from ``setup`` after deciding
        ``val.minimize`` there. Direction otherwise comes from ``val.minimize``,
        defaulting to maximizing.

        A trainer that never tracks a best never gets these attributes, so it stays
        free to use those names itself. Only ``best_model`` and ``best_metrics`` are
        always defined, since code outside this method reads them.
        """
        if minimize is None:
            minimize = bool(OmegaConf.select(self.cfg, "val.minimize", default=False))
        self.val_minimize = minimize
        patience = OmegaConf.select(self.cfg, "val.patience")
        if patience is None:
            # unset patience means never stop early, and num_epochs is that bound
            patience = OmegaConf.select(self.cfg, "num_epochs", default=float("inf"))
        self.patience = self._initial_patience = patience
        self.start_patience = OmegaConf.select(self.cfg, "val.start_patience", default=0)
        self.best_val = float("inf") if self.val_minimize else float("-inf")
        self.best_epoch = 0
        self.best_model = None
        self.best_metrics = {}
        self._best_tracking_initialized = True

    def save_if_best(self, score: float, metrics: dict | None = None) -> bool:
        """Keep this epoch's weights if its score is the best seen so far.

        On an improvement: records the score and epoch, snapshots the weights into
        ``best_model``, and writes ``best.pt``. The two have different readers:
        ``best_model`` is what ``test`` reloads in this same process, and it survives
        ``ckpt.enable=false``; ``best.pt`` is what a later run loads. Pass ``metrics``
        to also record the ``best/val/*`` dict for logging. Feed the result to
        :meth:`step_patience` to early-stop as well; a trainer that only wants
        best-model selection can call this alone.

        Call it on every rank, since every rank's ``test`` reloads its own
        ``best_model``; the ``best.pt`` write self-gates to rank 0 on its own. A trainer
        whose score only exists on rank 0 may call it under a rank guard, but then owes
        :meth:`step_patience` a verdict broadcast to the other ranks.

        Args:
            score: The scalar being tracked, in ``val.minimize`` direction. A
                non-finite score never counts as an improvement.
            metrics: The epoch's val metrics, if they should be recorded too.

        Returns:
            True if the score improved, i.e. the weights were kept.
        """
        if not self._best_tracking_initialized:
            self.reset_best_tracking()

        # NaN compares False against anything, so an unguarded NaN would read as an
        # improvement and then win every later comparison as the incumbent best
        if not math.isfinite(score):
            return False

        sign = -1 if self.val_minimize else 1
        if sign * score <= sign * self.best_val:
            return False

        if metrics is not None:
            self.best_val_metrics = {f"best/val/{k}": v for k, v in metrics.items()}
            self.best_val_metrics["best/val/avg"] = score
            self.best_metrics = self.best_val_metrics | {"best/epoch": self.epoch}
            self.logger.log_dict(self.best_metrics)
        self.best_val = score
        self.best_epoch = self.epoch
        # snapshot on the host: it is held for the whole run and only read at test time
        self.best_model = {
            k: v.detach().to("cpu", copy=True) if torch.is_tensor(v) else deepcopy(v)
            for k, v in self.model.state_dict().items()
        }
        self._save_best_checkpoint()
        return True

    def step_patience(self, improved: bool) -> bool:
        """Advance the early-stopping counter with this epoch's outcome.

        An improvement refills the counter; otherwise it ticks down, but only once
        the epoch reaches ``val.start_patience``.

        Every rank has to reach the same verdict, or one leaves the epoch loop while the
        others block in the next collective. That holds on its own when the score is
        rank-invariant, i.e. a torchmetrics ``compute()`` or a :meth:`reduce_mean`
        result; a rank-0-only score has to be broadcast by the caller instead.

        Args:
            improved: What :meth:`save_if_best` returned for this epoch.

        Returns:
            True once patience is exhausted, i.e. training should stop.
        """
        if not self._best_tracking_initialized:
            self.reset_best_tracking()

        if improved:
            self.patience = self._initial_patience
        elif self.epoch >= self.start_patience:
            self.patience -= 1

        if self.patience <= 0:
            self.logger.info(
                f"Early stopping triggered at epoch {self.epoch}, best epoch was {self.best_epoch}"
            )
            return True
        return False

    def get_param_groups(self, *modules: torch.nn.Module) -> list[dict]:
        """Split parameters into a weight-decayed and an undecayed optimizer group.

        A parameter skips weight decay if it is 1-D (biases, norm and other scale
        vectors) or if its name contains one of ``cfg.no_weight_decay``. The default list
        covers biases, norms and every ``*_emb`` lookup table; a model overrides it only
        when a name lies about what it is.

        Args:
            modules: Modules to take parameters from, ``self.model`` when none are
                given. Pass more when parameters outside the model are trained by the
                same optimizer, such as an objective with its own projection head.
        """
        return param_groups(
            *(modules or (self.model,)),
            weight_decay=self.cfg.weight_decay,
            no_weight_decay=self.cfg.get("no_weight_decay", DEFAULT_NO_WEIGHT_DECAY),
        )

    def log_lr(self):
        """Log the current learning rate for each optimizer parameter group."""
        for i, param_group in enumerate(self.optimizer.param_groups):
            name = param_group.get("name", f"group_{i}")
            self.logger.log(f"lr/{name}", param_group["lr"])

    def clip_and_log_grad_norm(self, *modules: torch.nn.Module) -> torch.Tensor:
        """Clip gradients in place and log the pre-clip norm as ``train/grad_norm``.

        The norm is measured even when ``grad_clip`` is unset so the metric does not
        vanish on unclipped runs. Call once per optimizer step, after ``backward()``.

        Args:
            modules: Modules to clip over, ``self.model`` by default. Pass the same ones
                as :meth:`get_param_groups`, so the norm spans every parameter stepped.
        """
        self.log_param_grad_stats(*modules)
        max_norm = self.cfg.get("grad_clip")
        grad_norm = torch.nn.utils.clip_grad_norm_(
            chain.from_iterable(m.parameters() for m in (modules or (self.model,))),
            max_norm=float("inf") if max_norm is None else max_norm,
        )
        if self.cfg.get("log_train_step", True):
            self.logger.log("train/grad_norm", grad_norm.item(), pbar=False)
        else:
            self._grad_norms.append(grad_norm.item())
        return grad_norm

    def log_epoch_grad_norm(self):
        """Log the epoch-mean grad norm, the counterpart to the epoch-mean train loss."""
        if self._grad_norms:
            self.logger.log("train/grad_norm", sum(self._grad_norms) / len(self._grad_norms))
            self._grad_norms.clear()

    def log_param_grad_stats(self, *modules: torch.nn.Module):
        """Log per-parameter weight norm, grad norm, and grad-to-weight ratio.

        Each gate costs one W&B key per parameter, so both are throttled to every
        ``log_stats_every_n_steps`` steps. Call after ``backward()``, before any clip.
        """
        weights, grads = self.cfg.get("log_weights"), self.cfg.get("log_grads")
        if not (weights or grads):
            return
        every_n = self.cfg.get("log_stats_every_n_steps", 1)
        if every_n <= 0 or self.train_step % every_n:
            return

        named = [
            (name.removeprefix("module."), param)  # keys match across DDP and single-GPU
            for module in (modules or (self.model,))
            for name, param in module.named_parameters()
            if param.requires_grad
        ]
        if not named:
            return

        # one host sync, not one per parameter
        zero = torch.zeros((), device=self.device)
        norms = torch.stack(
            [p.detach().norm().float() for _, p in named]
            + [zero if p.grad is None else p.grad.detach().norm().float() for _, p in named]
        ).tolist()
        w_norms, g_norms = norms[: len(named)], norms[len(named) :]

        stats = {}
        for (name, param), w, g in zip(named, w_norms, g_norms, strict=True):
            if weights:
                stats[f"weight_norm/{name}"] = w
            if grads and param.grad is not None:
                stats[f"grad_norm/{name}"] = g
                if w > 0:
                    stats[f"grad_weight_ratio/{name}"] = g / w
        if weights:
            stats["weight_norm/total"] = sum(w * w for w in w_norms) ** 0.5
        self.logger.log_dict(stats)

    def push_logs(self):
        """Flush accumulated logs to W&B, including current epoch and step."""
        self.logger.log("train/epoch", self.epoch)
        self.logger.log("train/step", self.train_step)
        self.logger.push()

    def get_best_pbar_metrics(self) -> dict:
        """Return the best validation metrics formatted for the epoch progress bar.

        Returns:
            dict: Best validation metrics with the ``best/val/`` prefix stripped,
                or an empty dict if no validation has been run yet.
        """
        if hasattr(self, "best_val_metrics"):
            return {k.replace("best/val/", ""): v for k, v in self.best_val_metrics.items()}
        return {}
