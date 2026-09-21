import multiprocessing as mp

import hydra
import torch
import torch.distributed as dist
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from torch.utils.data import DataLoader, RandomSampler

from core.dataset import WholeSessionSpikeDataset
from core.nn.loss import CLIPLoss
from core.samplers import DistributedSamplerWrapper
from core.trainer import BaseTrainer
from core.utils.util import get_num_params, get_num_trainable_params, human_readable

from .augmentations import get_acg_transform, get_wf_transform
from .cache import NEMO_UNIT_QC
from .dataset import NEMOAugmentedTensorDataset
from .encoding import encode, load_units, preprocess
from .monitor import BrainRegionMonitor
from .nemo import NEMO


class NEMOPretrain(BaseTrainer):
    """Trainer for NEMO SSL pretraining (CLIP-style contrastive learning)."""

    def setup(self, ckpt: dict | None):
        self.logger.info(f"Trainer: {self.__class__}")

        self._setup_data()
        self.setup_model()
        self.setup_train_loader()
        self.setup_optimizer(ckpt)
        if self.rank == 0:
            self.monitor: BrainRegionMonitor | None = None
            if self.cfg.monitor is not None:
                self.monitor = hydra.utils.instantiate(self.cfg.monitor, world_size=self.world_size)

    def train_epoch(self):
        for wf_batch, acg_batch in self.logger.get_pbar(self.train_loader, prefix="train"):
            wf_batch, acg_batch = preprocess(wf_batch, acg_batch, self.device)
            self.optimizer.zero_grad()

            z_wf, z_acg = self.model(wf_batch, acg_batch.unsqueeze(1))
            loss = self.loss_fn(z_wf, z_acg)
            loss.backward()
            self.clip_and_log_grad_norm()
            self.optimizer.step()
            self.lr_scheduler.step()
            self.logger.log("train/loss", loss.item(), pbar=True)
            self.train_step += 1
        self.push_logs()

    @torch.inference_mode()
    def val_epoch(self):
        self.model.eval()

        # no monitor means no score, hence no best-model selection and no early stopping
        early_stop = False
        if self.rank == 0 and self.monitor is not None:
            train_embs = encode(
                self.model,
                self.train_waveforms,
                self.train_acgs,
                self.cfg.batch_size,
                self.device,
                self.precision.dtype,
                self.logger,
            )
            metrics = self.monitor(train_embs=train_embs, train_uids=self.train_uids)
            self.logger.log_dict(metrics)
            self.return_value = {**metrics, "val/epoch": self.epoch}
            self.logger.log("val/epoch", self.epoch)

            improved = self.save_if_best(metrics[self.cfg.val.metric])
            early_stop = self.step_patience(improved)

        self.push_logs()

        # only rank 0 scores, but every rank must leave the training loop together
        if self.is_distributed:
            flag = torch.tensor(int(early_stop), device=self.device)
            dist.broadcast(flag, src=0)
            early_stop = bool(flag.item())
        return early_stop

    def _setup_data(self):
        dataset = WholeSessionSpikeDataset(
            root=self.cfg.data_root, regime="pretrain", unit_qc=NEMO_UNIT_QC
        )
        self.train_waveforms, self.train_acgs, self.train_uids = load_units(
            self.cfg.cache_path, dataset
        )

    def setup_model(self):
        self.model = hydra.utils.instantiate(self.cfg.model)
        if not isinstance(self.model, NEMO):
            raise TypeError(
                f"{type(self).__name__} requires a NEMO model, got {type(self.model).__name__}"
            )
        self.add_checkpoint_items(model=self.model)
        self.loss_fn = CLIPLoss(temperature=self.cfg.loss.temperature)

        self.logger.info(f"Precision: {self.precision}")
        self.logger.info(f"Model: {self.model.__class__}")

        self.model.train()

        num_params, skipped_params = get_num_params(self.model)
        num_params_acg, _ = get_num_params(self.model.acg_encoder)
        num_params_wvf, _ = get_num_params(self.model.wvf_encoder)
        num_params_acg_proj, _ = get_num_params(self.model.acg_projector)
        num_params_wvf_proj, _ = get_num_params(self.model.wvf_projector)
        num_trainable_params = get_num_trainable_params(self.model, return_skipped=False)
        self.logger.info(f"Number of parameters: {human_readable(num_params)} ({num_params:,})")
        self.logger.info(
            f"Number of ACG encoder params: {human_readable(num_params_acg)} ({num_params_acg:,})"
        )
        self.logger.info(
            f"Number of wvf encoder params: {human_readable(num_params_wvf)} ({num_params_wvf:,})"
        )
        self.logger.info(
            f"Number of ACG projector params: {human_readable(num_params_acg_proj)} ({num_params_acg_proj:,})"
        )
        self.logger.info(
            f"Number of wvf projector params: {human_readable(num_params_wvf_proj)} ({num_params_wvf_proj:,})"
        )
        self.logger.info(
            f"Number of trainable parameters: {human_readable(num_trainable_params)} ({num_trainable_params:,})"
        )
        if skipped_params:
            self.logger.info(f"Skipped lazy parameters: {skipped_params}")

        self.model = self.make_ddp(self.model.to(self.device))

    def setup_train_loader(self):
        has_workers = self.cfg.num_workers > 0
        dataset = NEMOAugmentedTensorDataset(
            self.train_waveforms,
            self.train_acgs,
            wvf_transform=get_wf_transform(),
            acg_transform=get_acg_transform(),
        )
        self.train_sampler = DistributedSamplerWrapper(
            RandomSampler(dataset, generator=torch.Generator().manual_seed(self.cfg.seed))
        )
        self.train_loader = DataLoader(
            dataset,
            sampler=self.train_sampler,
            batch_size=self.cfg.batch_size // self.world_size,
            num_workers=self.cfg.num_workers,
            pin_memory=self.cfg.pin_memory,
            prefetch_factor=self.cfg.prefetch_factor if has_workers else None,
            persistent_workers=self.cfg.persistent_workers if has_workers else False,
            multiprocessing_context=mp.get_context("fork") if has_workers else None,
        )

    def setup_optimizer(self, ckpt: dict | None):
        if self.model is None:
            raise ValueError("Trying to set up optimizers before the model is setup")
        if self.train_loader is None:
            raise ValueError("Trying to set up optimizers before the train data loader is setup")

        grouped_parameters = self.get_param_groups()

        self.optimizer = torch.optim.AdamW(
            grouped_parameters,
            lr=self.cfg.base_lr,
        )

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

        self.add_checkpoint_items(
            optimizer=self.optimizer,
            lr_scheduler=self.lr_scheduler,
        )
