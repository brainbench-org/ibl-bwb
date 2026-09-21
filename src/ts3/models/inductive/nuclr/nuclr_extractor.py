"""NuCLR's unit-embedding extractor.

Inductive: one frozen checkpoint answers both regimes, because a NuCLR unit embedding is a
function of that unit's spikes and nothing about the unit's identity is stored in the
weights. Averaging a sequence model's output over the windows a unit appears in is what
turns it into a per-unit embedding, and the window length, the 50% overlap and the one-view
collate are the model's, read off the config it trained with.
"""

import multiprocessing as mp
from functools import partial

import hydra
import numpy as np
import torch
from torch_brain.samplers import SequentialFixedWindowSampler

from core.dataset import BenchmarkRegime
from core.utils.embedding_collector import EmbeddingCollector
from core.utils.logger import Logger
from core.utils.util import Precision, expand_path, move_to_device
from pretrain.models.nuclr.dataset import NuCLRDataset
from ts3.models.base import Extractor, load_pretrained


class NuCLRExtractor(Extractor):
    def __init__(self, ckpt: str, batch_size: int | None = None, num_workers: int | None = None):
        self.ckpt = expand_path(ckpt).resolve()
        self.batch_size = batch_size
        self.num_workers = num_workers

    @property
    def name(self) -> str:
        return f"{self.ckpt.parent.name}_{self.ckpt.stem}"

    @property
    def run_seed(self) -> int | None:
        return self.cfg.get("seed")

    def setup(self, data_root, device: torch.device, logger: Logger) -> None:
        super().setup(data_root, device, logger)
        self.model, self.cfg = load_pretrained(
            self.ckpt,
            device,
            logger,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
        )

    def _build_loader(self, regime: BenchmarkRegime):
        """The sequential one-view loader, stepping half a context at a time.

        Unlike the training loader it keeps the final short batch, there being no reason for
        an inference pass to discard data. Some units still end with no embedding: the
        collector is sized to every unit in the dataset, while ``eval_transform`` and the
        qc filter remove units from the batches. The probe join drops those.
        """
        ds = NuCLRDataset(root=self.data_root, regime=regime, input_fn=self.model.input_fn)
        if self.cfg.eval_transform:
            ds.transform = hydra.utils.instantiate(self.cfg.eval_transform)

        sampler = SequentialFixedWindowSampler(
            sampling_intervals=ds.get_sampling_intervals(),
            window_length=self.model.ctx_duration,
            step=0.5 * self.model.ctx_duration,
            drop_short=True,
        )
        has_workers = self.cfg.num_workers > 0
        loader = torch.utils.data.DataLoader(
            dataset=ds,
            sampler=sampler,
            collate_fn=partial(self.model.collate, two_view=False),
            batch_size=self.cfg.batch_size,
            num_workers=self.cfg.num_workers,
            drop_last=False,
            pin_memory=self.cfg.pin_memory,
            multiprocessing_context=mp.get_context("fork") if has_workers else None,
        )
        return ds, loader

    @torch.inference_mode()
    def encode(self, regime: BenchmarkRegime) -> tuple[torch.Tensor, np.ndarray]:
        """One mean embedding per unit, over every window that unit appears in."""
        self.model.eval()
        ds, loader = self._build_loader(regime)
        self.logger.info(
            f"{regime}: {len(ds.get_session_ids())} sessions"
            f", {len(ds.get_unit_ids())} units"
            f", {len(loader)} batches"
        )

        collector = EmbeddingCollector(
            unit_ids=ds.get_unit_ids(),
            dim=self.model.emb_dim,
            device=self.device,
        )
        dtype = Precision(self.cfg.precision).dtype
        for X in self.logger.get_pbar(loader, prefix=f"emb-{regime}"):
            if X is None:
                # Some sessions have all bad neurons, and sequential sampling can land a
                # whole batch inside them.
                continue
            X = move_to_device(X, self.device)
            with torch.autocast(device_type=self.device.type, dtype=dtype):
                z = self.model(**X["model_inputs"])
            assert torch.isfinite(z).all()
            collector.update(z, X["unit_ids"])

        return collector.compute()
