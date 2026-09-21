"""Calibration: continue MtM's multi-task masked-reconstruction objective on one recording.

This trainer is used for TS3 transductive evaluation for MtM, wherein we generate a brain
region prediction by first calibrating the model on the new session, creating an
``in_stitcher`` / ``out_stitcher`` pair for that session's channels by backprop. This
produces a session-local checkpoint whose stitcher columns
:class:`~ts3.models.transductive.MtMExtractor` reads as the eval part of an embeddings file,
which a probe in ``ts3/eval.py`` then classifies into brain regions.

This trainer directly extends the MtM pretrain trainer
:class:`~pretrain.models.mtm.MtMPretrain`. Three things have to be added on top of it.
``setup_model`` instantiates with ``finetune_enable=True``. Its loaders hardcode
``regime="pretrain"`` and a ``pretrain_{split}_domain`` no eval recording carries, so
``setup_train_loader`` / ``setup_val_loader`` build from
:class:`~ts3.models.transductive.calibration_dataset.TrialAlignedCalibrationDataset` instead, one recording split causally in time.
``link_model`` calls ``MtM.load_ckpt``: the pretrain trainer accepts a ``ckpt`` argument and
currently ignores it, being only ever run once from scratch, so without this the pretrained backbone
would never be read and the fresh stitchers would have nothing to attach to.
"""

import multiprocessing as mp

import hydra
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch_brain.batching import collate
from torch_brain.samplers import RandomFixedWindowSampler, SequentialFixedWindowSampler
from torch_brain.transforms import Compose

from core.samplers import DistributedSamplerWrapper, SessionBatchSampler
from core.utils.util import log_param_breakdown
from pretrain.models.mtm import MtM, MtMMasker, MtMPretrain
from ts3.models.transductive.calibration_dataset import TrialAlignedCalibrationDataset


class MtMCalibrateTrainer(MtMPretrain):
    def setup_model(self, ckpt: dict | None = None):
        self.model = hydra.utils.instantiate(self.cfg.model, finetune_enable=True)
        if not isinstance(self.model, MtM):
            raise TypeError(
                f"{type(self).__name__} requires a MtM model, got {type(self.model).__name__}"
            )
        self.masker = hydra.utils.instantiate(self.cfg.masker)
        if not isinstance(self.masker, MtMMasker):
            raise TypeError(
                f"{type(self).__name__} requires a MtMMasker masker, got {type(self.masker).__name__}"
            )
        self.add_checkpoint_items(model=self.model, masker=self.masker)

        self.logger.info(f"Precision: {self.precision}")
        self.logger.info(f"Model: {self.model.__class__}")
        self.logger.info(f"Masker: {self.masker.__class__}")
        self.model.train()

    def _make_loader(self, dataset, batch_sampler):
        has_workers = self.cfg.num_workers > 0
        return DataLoader(
            dataset,
            batch_sampler=batch_sampler,
            collate_fn=collate,
            num_workers=self.cfg.num_workers,
            pin_memory=self.cfg.pin_memory,
            prefetch_factor=self.cfg.prefetch_factor if has_workers else None,
            persistent_workers=self.cfg.persistent_workers if has_workers else False,
            multiprocessing_context=mp.get_context("fork") if has_workers else None,
        )

    def setup_train_loader(self):
        train_dataset = TrialAlignedCalibrationDataset(self.cfg.data_root, self.cfg.recording_ids)
        self._train_intervals, self._val_intervals = train_dataset.causal_split()

        batch_sampler = DistributedSamplerWrapper(
            SessionBatchSampler(
                RandomFixedWindowSampler(
                    sampling_intervals=self._train_intervals,
                    window_length=train_dataset.context_window,
                    generator=torch.Generator().manual_seed(self.cfg.seed),
                    drop_short=True,
                ),
                batch_size=self.cfg.batch_size // self.world_size,
                shuffle_batches=True,
                generator=torch.Generator().manual_seed(self.cfg.seed),
            )
        )
        self.train_dataset = train_dataset
        self.train_loader = self._make_loader(train_dataset, batch_sampler)
        self.logger.info(
            f"Calibration recording: {train_dataset.recording_ids[0]} "
            f"({len(train_dataset.get_unit_ids())} units)"
        )
        self.logger.info(f"Training on {len(self.train_loader)} batches")

    def setup_val_loader(self):
        val_dataset = TrialAlignedCalibrationDataset(self.cfg.data_root, self.cfg.recording_ids)
        batch_sampler = DistributedSamplerWrapper(
            SessionBatchSampler(
                SequentialFixedWindowSampler(
                    sampling_intervals=self._val_intervals,
                    window_length=val_dataset.context_window,
                    drop_short=True,
                ),
                batch_size=self.cfg.batch_size // self.world_size,
                drop_last=False,
            )
        )
        self.val_dataset = val_dataset
        self.val_loader = self._make_loader(val_dataset, batch_sampler)
        self.logger.info(f"Validating on {len(self.val_loader)} batches")

    def link_model(self, model: nn.Module, ckpt: dict | None = None):
        assert ckpt is not None, (
            "calibration requires ckpt.load_from to point at the shared pretrain checkpoint"
        )
        for split in ["train", "val"]:
            model_transforms = hydra.utils.instantiate(self.cfg.get(f"{split}_transforms", []))
            self.logger.info(f"Model transforms ({split}): {model_transforms}")
            dataset = getattr(self, f"{split}_loader").dataset

            if dataset.transform is None:
                dataset.transform = Compose([*model_transforms, model.input_fn])
            else:
                dataset.transform = Compose([dataset.transform, *model_transforms, model.input_fn])

        model.link_datasets(self.train_loader.dataset, self.val_loader.dataset)
        model.load_ckpt(ckpt)

        log_param_breakdown(self.logger, self.model, ("in_stitcher", "out_stitcher"))
