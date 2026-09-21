"""Calibration: continue POSSM's multi-task decoding objective on one recording.

This trainer is used for TS3 transductive evaluation for POSSM, wherein we generate a brain
region prediction by first calibrating the model on the new session, creating that session's
``unit_emb`` rows by backprop. This produces a session-local checkpoint whose unit table
:class:`~ts3.models.transductive.POSSMExtractor` reads as the eval part of an embeddings file,
which a probe in ``ts3/eval.py`` then classifies into brain regions.

This trainer directly extends the POSSM pretrain trainer
:class:`~pretrain.models.possm.POSSMMultitaskPretrain`. Four things have to be added on top
of it. ``setup_model`` instantiates with ``finetune_enable=True``, which POSSM takes as a
constructor argument rather than a ``link_datasets`` one. Its loaders hardcode
``regime="pretrain"`` and a ``pretrain_{split}_domain`` no eval recording carries, so
``setup_train_loader`` / ``setup_val_loader`` build from
:class:`~ts3.models.transductive.calibration_dataset.TrialAlignedCalibrationDataset` instead, one recording split causally in time.
``_restore_model_from_ckpt`` requires the pretrain checkpoint, since a calibration that
started from scratch would be a single-session model rather than an adapted one.
``cfg.finetuning`` is read by ts1/ts2's eval trainers and by no pretrain trainer, so
``setup`` and ``train_epoch`` apply it here: freshly initialised rows sit in front of a
pretrained backbone, and deferring the backbone's gradients is what
:class:`~core.finetuning.GradualUnfreezing` is for.
"""

import multiprocessing as mp

import hydra
import torch
from omegaconf import OmegaConf
from torch.utils.data import DataLoader
from torch_brain.batching import collate
from torch_brain.samplers import RandomFixedWindowSampler, SequentialFixedWindowSampler
from torch_brain.transforms import Compose

from core.samplers import DistributedSamplerWrapper
from pretrain.models.possm import POSSM, POSSMMultitaskPretrain
from ts3.models.transductive.calibration_dataset import TrialAlignedCalibrationDataset


class POSSMCalibrateTrainer(POSSMMultitaskPretrain):
    def setup_model(self):
        self.model = hydra.utils.instantiate(self.cfg.model, finetune_enable=True)
        if not isinstance(self.model, POSSM):
            raise TypeError(
                f"{type(self).__name__} requires a POSSM model, got {type(self.model).__name__}"
            )
        self.logger.info(f"Precision: {self.precision}")
        self.logger.info(f"Model: {self.model.__class__}")
        self.model.train()

    def _make_loader(self, dataset, sampler):
        has_workers = self.cfg.num_workers > 0
        return DataLoader(
            dataset,
            sampler=sampler,
            collate_fn=collate,
            batch_size=self.cfg.batch_size // self.world_size,
            num_workers=self.cfg.num_workers,
            drop_last=False,
            pin_memory=self.cfg.pin_memory,
            prefetch_factor=self.cfg.prefetch_factor if has_workers else None,
            persistent_workers=self.cfg.persistent_workers if has_workers else False,
            multiprocessing_context=mp.get_context("fork") if has_workers else None,
        )

    def setup_train_loader(self):
        train_dataset = TrialAlignedCalibrationDataset(
            self.cfg.data_root, self.cfg.recording_ids, tasks=self.tasks
        )
        self._train_intervals, self._val_intervals = train_dataset.causal_split()

        sampler = DistributedSamplerWrapper(
            RandomFixedWindowSampler(
                sampling_intervals=self._train_intervals,
                window_length=train_dataset.context_window,
                generator=torch.Generator().manual_seed(self.cfg.seed),
                drop_short=True,
            )
        )
        self.train_loader = self._make_loader(train_dataset, sampler)
        self.logger.info(
            f"Calibration recording: {train_dataset.recording_ids[0]} "
            f"({len(train_dataset.get_unit_ids())} units)"
        )
        self.logger.info(f"Training on {len(self.train_loader)} batches")

    def setup_val_loader(self):
        val_dataset = TrialAlignedCalibrationDataset(
            self.cfg.data_root, self.cfg.recording_ids, tasks=self.tasks
        )
        sampler = DistributedSamplerWrapper(
            SequentialFixedWindowSampler(
                sampling_intervals=self._val_intervals,
                window_length=val_dataset.context_window,
                drop_short=True,
            )
        )
        self.val_loader = self._make_loader(val_dataset, sampler)
        self.logger.info(f"Validating on {len(self.val_loader)} batches")

    def link_model(self, model: POSSM):
        for split in ["train", "val"]:
            model_transforms = hydra.utils.instantiate(self.cfg.get(f"{split}_transforms", []))
            self.logger.info(f"Model transforms ({split}): {model_transforms}")
            dataset = getattr(self, f"{split}_loader").dataset

            if dataset.transform is None:
                dataset.transform = Compose([*model_transforms, model.input_fn])
            else:
                dataset.transform = Compose([dataset.transform, *model_transforms, model.input_fn])

        model.link_datasets(self.train_loader.dataset, self.val_loader.dataset)
        model.configure_multitask_readout(self.readout_specs)

    def _restore_model_from_ckpt(self, ckpt: dict | None):
        assert ckpt is not None, (
            "calibration requires ckpt.load_from to point at the shared pretrain checkpoint"
        )
        self.model.load_ckpt(ckpt)

    def setup(self, ckpt: dict | None):
        """The parent's setup, plus the finetuning strategy the eval trainers apply.

        ``cfg.finetuning`` is read by ts1/ts2's eval trainers but not by the pretrain trainer,
        so calibration inheriting the pretrain one would accept the block and ignore it.
        """
        super().setup(ckpt)
        self.setup_finetuning()

    def setup_finetuning(self):
        self.ft_strategy = None
        strategy = OmegaConf.select(self.cfg, "finetuning.strategy")
        if strategy is None:
            return
        self.ft_strategy = hydra.utils.instantiate(
            strategy, model=self.model, cfg=self.cfg, _recursive_=False
        )
        if self.ft_strategy.enable:
            self.ft_strategy.setup()

    def train_epoch(self):
        if getattr(self, "ft_strategy", None) is not None and self.ft_strategy.enable:
            self.ft_strategy.update(self.epoch)
        return super().train_epoch()
