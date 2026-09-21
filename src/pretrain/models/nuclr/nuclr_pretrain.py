import multiprocessing as mp
from functools import partial

import hydra
import torch
from omegaconf import DictConfig
from torch_brain.samplers import RandomFixedWindowSampler

from core.samplers import DistributedSamplerWrapper
from core.trainer import BaseTrainer
from core.utils.embedding_collector import EmbeddingCollector
from core.utils.util import (
    get_num_params,
    human_readable,
    move_to_device,
    rank_zero_only,
)

from .dataset import NuCLRDataset
from .monitor import BrainRegionMonitor
from .nuclr import NuCLR


class NuCLRPretrain(BaseTrainer):
    """Contrastive pretraining of the NuCLR unit encoder, against :class:`NuCLRLoss`."""

    def setup(self, ckpt: dict | None):
        self.logger.info(f"Trainer: {self.__class__}")

        if ckpt is not None:
            self.ckpt_cfg = DictConfig(ckpt["cfg"])

        self.setup_model(ckpt)
        self.setup_ssl_loss(ckpt)
        self.setup_train_loader()
        self.setup_monitor()
        self.setup_optimizer(ckpt)

        self.train_collector = EmbeddingCollector(
            unit_ids=self.train_dataset.get_unit_ids(),
            dim=self.model.emb_dim,  # ty:ignore[invalid-argument-type]
            device=self.device,
        )

        self.model = self.make_ddp(self.model)
        self.ssl_loss = self.make_ddp(self.ssl_loss)

    def train_epoch(self):
        self.model.train()
        self.ssl_loss.train()
        self.train_collector.reset()

        for X in self.logger.get_pbar(self.train_loader, prefix="train"):
            with torch.autocast(device_type="cuda", dtype=self.precision.dtype):
                X = move_to_device(X, self.device)
                z = self.model(**X["model_inputs"])
                assert torch.isfinite(z).all()

                loss, loss_dict = self.ssl_loss(
                    x=z,
                    seqlen=X["unit_seqlen"],
                    unit_ids=X["unit_ids"],
                    probe_ids=X["probe_ids"],
                    prefix="train/",
                )
                assert torch.isfinite(loss).all()

            self.train_collector.update(z, X["unit_ids"])

            loss.backward()
            self.clip_and_log_grad_norm(self.model, self.ssl_loss)
            self.optimizer.step()
            self.lr_scheduler.step()
            self.optimizer.zero_grad()

            self.logger.log_dict(loss_dict)
            self.logger.log("Loss", loss.item(), pbar=True, wandb=False)
            self.log_lr()
            self.push_logs()
            self.train_step += 1

    @torch.inference_mode()
    def val_epoch(self):
        self.model.eval()
        self.ssl_loss.eval()

        # the embeddings the training loop already collected, probed on pretrain units
        train_embs, train_uids = self.train_collector.compute()

        if self.rank == 0 and self.monitor is not None:
            metrics = self.monitor(train_embs=train_embs, train_uids=train_uids)
            self.logger.log_dict(metrics)

    def setup_model(self, ckpt: dict | None):
        self.logger.info(f"Precision: {self.precision}")

        if ckpt is not None:
            self.cfg.model = self.ckpt_cfg.model
            self.logger.info("Model: Overriding config from ckpt")
            self.logger.info(self.cfg.model)
        else:
            self.cfg.model.ctx_duration = self.cfg.views.duration
        self.model: torch.nn.Module = hydra.utils.instantiate(self.cfg.model)
        if not isinstance(self.model, NuCLR):
            raise TypeError(
                f"{type(self).__name__} requires a NuCLR model, got {type(self.model).__name__}"
            )

        num_params, _ = get_num_params(self.model)
        self.logger.info(f"Model: {self.model.__class__}")
        self.logger.info(f"Model: params - {human_readable(num_params)}")

        if ckpt is not None:
            self.model.load_state_dict(ckpt["model_state_dict"])
            self.logger.info("Model: loaded state from checkpoint")

        self.model = self.model.to(self.device)
        self.model.train()
        self.add_checkpoint_items(model=self.model)

    def setup_ssl_loss(self, ckpt: dict | None):
        if ckpt is not None and not self.cfg.ckpt.model_only:
            self.cfg.loss = self.ckpt_cfg.loss
            self.logger.info("Loss: Overriding config from ckpt")
            self.logger.info(self.cfg.loss)

        self.ssl_loss: torch.nn.Module = hydra.utils.instantiate(
            self.cfg.loss,
            dim_in=self.model.emb_dim,
        )
        self.logger.info(f"Loss: {self.ssl_loss.__class__}")

        num_params, _ = get_num_params(self.ssl_loss)
        self.logger.info(f"Loss: params - {human_readable(num_params)}")

        if ckpt is not None:
            if not self.cfg.ckpt.model_only:
                self.ssl_loss.load_state_dict(ckpt["ssl_loss_state_dict"])
                self.logger.info("Loss: loaded state from checkpoint")
            else:
                self.logger.info("Loss: checkpoint not loaded due to ckpt.model_only")

        self.ssl_loss = self.ssl_loss.to(self.device)
        self.ssl_loss.train()
        self.add_checkpoint_items(ssl_loss=self.ssl_loss)

    def setup_optimizer(self, ckpt: dict | None):
        from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR

        # the contrastive loss carries its own projector, trained by the same optimizer
        grouped_parameters = self.get_param_groups(self.model, self.ssl_loss)

        self.optimizer = torch.optim.AdamW(
            grouped_parameters,
            lr=self.cfg.base_lr,
        )
        self.logger.info(f"Optimizer: {self.optimizer.__class__}")

        steps_per_epoch = len(self.train_loader)
        warmup_steps = 1 * steps_per_epoch
        total_steps = steps_per_epoch * self.cfg.num_epochs
        s1 = LinearLR(self.optimizer, start_factor=0.01, total_iters=warmup_steps)
        s2 = CosineAnnealingLR(
            self.optimizer,
            T_max=total_steps - warmup_steps,
            eta_min=self.cfg.base_lr * self.cfg.lr_decay,
        )
        self.lr_scheduler = SequentialLR(
            self.optimizer,
            schedulers=[s1, s2],
            milestones=[warmup_steps],
        )
        self.logger.info(f"LR Scheduler: {self.lr_scheduler}")

        # both optim and scheduler states needed for resuming
        self.add_checkpoint_items(
            optimizer=self.optimizer,
            lr_scheduler=self.lr_scheduler,
        )

    def setup_train_loader(self):
        ds = NuCLRDataset(
            root=self.cfg.data_root,
            regime="pretrain",
            input_fn=self.model.input_fn,  # ty:ignore[invalid-argument-type]
            context_length=self.cfg.views.duration,
        )
        ds.enable_two_views(
            max_distance=self.cfg.views.max_distance,
            seed=self.cfg.seed,
        )
        if self.cfg.train_transform:
            ds.transform = hydra.utils.instantiate(self.cfg.train_transform)

        sampler = DistributedSamplerWrapper(
            RandomFixedWindowSampler(
                sampling_intervals=ds.get_sampling_intervals(),
                window_length=ds.context_window,
                generator=torch.Generator().manual_seed(self.cfg.seed),
            )
        )

        collate = partial(self.model.collate, two_view=True)  # type: ignore

        has_workers = self.cfg.num_workers > 0
        loader = torch.utils.data.DataLoader(
            dataset=ds,
            sampler=sampler,
            collate_fn=collate,
            batch_size=self.cfg.batch_size // self.world_size,
            num_workers=self.cfg.num_workers,
            drop_last=True,
            pin_memory=self.cfg.pin_memory,
            prefetch_factor=self.cfg.prefetch_factor if has_workers else None,
            persistent_workers=self.cfg.persistent_workers if has_workers else False,
            multiprocessing_context=mp.get_context("fork") if has_workers else None,
        )
        self.logger.info(
            f"Pretrain dataset: {len(ds.get_session_ids())} sessions"
            f", {len(ds.get_unit_ids())} units"
            f", {len(loader)} batches per epoch."
        )

        self.train_dataset = ds
        self.train_loader = loader

    @rank_zero_only
    def setup_monitor(self):
        self.monitor: BrainRegionMonitor | None = None
        if self.cfg.monitor is not None:
            self.monitor = hydra.utils.instantiate(self.cfg.monitor, world_size=self.world_size)
