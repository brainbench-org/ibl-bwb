"""POSSM's unit-embedding extractor.

A unit embedding is its row of ``unit_emb``, present only for units the checkpoint was
trained on, so the eval units of a held-out recording have no row until calibration
creates one.
"""

import numpy as np
import torch

from core.dataset import BenchmarkRegime, WholeSessionSpikeDataset
from ts3.models.transductive.base import TransductiveExtractor


class POSSMExtractor(TransductiveExtractor):
    """POSSM's ``unit_emb`` rows, optionally length-matched to the pretrain part.

    ``normalize="scale_only"`` multiplies the eval part by one scalar so its mean vector
    length matches the pretrain part's. Every eval vector keeps its direction and the eval
    cloud keeps its shape; only its radius moves.
    """

    def __init__(self, *args, normalize: str = "none", **kwargs):
        super().__init__(*args, **kwargs)
        assert normalize in ("none", "scale_only"), f"unknown normalize: {normalize}"
        self.normalize = normalize
        self._train_norm: float | None = None

    @property
    def name(self) -> str:
        return super().name + ("" if self.normalize == "none" else f"_{self.normalize}")

    def encode(self, regime: BenchmarkRegime) -> tuple[torch.Tensor, np.ndarray]:
        embs, uids = super().encode(regime)

        if regime == "pretrain":
            self._train_norm = float(embs.float().norm(dim=1).mean())
            return embs, uids

        if self.normalize == "scale_only":
            assert self._train_norm is not None, (
                "scale_only reads the pretrain part's norm, so encode('pretrain') runs first"
            )
            norm = float(embs.float().norm(dim=1).mean())
            if norm == 0.0:
                self.logger.warning("eval embeddings are all zero, not rescaling")
            else:
                embs = embs * (self._train_norm / norm)
                self.logger.info(
                    f"scale_only: eval mean ||x|| {norm:.3f} -> {self._train_norm:.3f}"
                )

        return embs, uids

    def read(
        self, state_dict: dict, dataset: WholeSessionSpikeDataset, uids: set[str]
    ) -> tuple[torch.Tensor, np.ndarray]:
        weights = state_dict["unit_emb.weight"]  # [V, D]
        vocab = state_dict["unit_emb.vocab"]  # OrderedDict[uid, idx]
        assert len(vocab) == len(weights), (
            f"vocab size {len(vocab)} != embedding rows {len(weights)}"
        )

        # one ordering, used for both the lookup and the uids, so they cannot disagree
        out_uids = sorted(uids)
        missing = [u for u in out_uids if u not in vocab]
        assert not missing, f"{len(missing)} uids not found in vocab, e.g. {missing[:5]}"

        idx = torch.tensor([vocab[u] for u in out_uids], dtype=torch.long)
        return weights[idx].contiguous(), np.asarray(out_uids).astype(str)
