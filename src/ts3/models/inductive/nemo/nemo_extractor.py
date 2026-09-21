"""NEMO's unit-embedding extractor.

Inductive: one frozen checkpoint answers both regimes, because a NEMO unit embedding is a
function of that unit's waveform and autocorrelogram. The encoding contract itself
(peak-normalisation, the ACG scale factor, the ``[acg, wf]`` concat order) is the model's and
stays in ``pretrain.models.nemo.encoding``, shared with the online monitor; what is TS3's
here is only choosing a regime and reading the cache for it.
"""

import numpy as np
import torch

from core.dataset import BenchmarkRegime
from core.utils.logger import Logger
from core.utils.util import Precision, expand_path
from pretrain.models.nemo.encoding import encode as encode_units
from pretrain.models.nemo.encoding import load_units
from ts3.models.base import Extractor, load_pretrained
from ts3.ts3_dataset import IBLBrainWideBenchTS3


class NEMOExtractor(Extractor):
    def __init__(self, ckpt: str, batch_size: int | None = None):
        self.ckpt = expand_path(ckpt).resolve()
        self.batch_size = batch_size

    @property
    def name(self) -> str:
        return f"{self.ckpt.parent.name}_{self.ckpt.stem}"

    @property
    def run_seed(self) -> int | None:
        return self.cfg.get("seed")

    def setup(self, data_root, device: torch.device, logger: Logger) -> None:
        super().setup(data_root, device, logger)
        self.model, self.cfg = load_pretrained(
            self.ckpt, device, logger, batch_size=self.batch_size
        )

    def encode(self, regime: BenchmarkRegime) -> tuple[torch.Tensor, np.ndarray]:
        """One embedding per unit, from its waveform and autocorrelogram."""
        dataset = IBLBrainWideBenchTS3(root=self.data_root, regime=regime)
        waveforms, acgs, uids = load_units(self.cfg.cache_path, dataset)
        self.logger.info(f"{regime}: {len(uids)} units")
        embs = encode_units(
            self.model,
            waveforms,
            acgs,
            self.cfg.batch_size,
            self.device,
            Precision(self.cfg.precision).dtype,
            self.logger,
        )
        return embs, uids
