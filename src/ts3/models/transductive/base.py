"""What every transductive extractor shares: which checkpoint answers which regime.

The train side comes from the shared pretrain checkpoint and the eval side from one
finetuning checkpoint per eval recording, so the eval part of the embeddings file is a
concatenation across encoders that drifted apart on their own sessions. That is the regime,
not a defect, but it is why the two halves are less comparable here than under
``ts3.models.inductive``, and why the probe is the only thing that ever sees them together.

Subclasses supply :meth:`read`, which turns one checkpoint into embeddings for the units it
was trained on. They exist as a family because for these models a unit embedding is a free
parameter indexed by unit identity: no row exists for a unit the checkpoint never saw, so
there is nothing to read for held-out units without finetuning on them first.
"""

from abc import abstractmethod

import numpy as np
import torch
from tqdm import tqdm

from core.dataset import BenchmarkRegime, WholeSessionSpikeDataset
from core.utils.logger import Logger
from core.utils.util import expand_path
from ts3.models.base import Extractor
from ts3.models.transductive.ckpt_index import resolve_finetune_ckpts


class TransductiveExtractor(Extractor):
    def __init__(
        self,
        pretrain_ckpt: str,
        seed: int,
        ckpt_dir: str,
        filters: dict | None = None,
        csv_path: str | None = None,
    ):
        self.pretrain_ckpt = expand_path(pretrain_ckpt).resolve()
        self.seed = seed
        self.ckpt_dir = ckpt_dir
        self.filters = filters
        self.csv_path = csv_path

    @property
    def name(self) -> str:
        return f"{self.pretrain_ckpt.parent.name}_{self.pretrain_ckpt.stem}_s{self.seed}"

    @property
    def run_seed(self) -> int | None:
        # the finetuning seed: the pretrain checkpoint is shared, so this is what varies
        return self.seed

    def setup(self, data_root, device: torch.device, logger: Logger) -> None:
        super().setup(data_root, device, logger)
        # failing units are kept here and filtered later, at eval, so nothing is cut twice.
        # That is a population TS3 does not score, so it does not come through its dataset.
        self.datasets = {
            regime: WholeSessionSpikeDataset(
                root=data_root,
                regime=regime,
                contract="TS3 transductive",
            )
            for regime in ("pretrain", "eval")
        }
        self.uids = {r: set(ds.get_unit_ids()) for r, ds in self.datasets.items()}
        assert not (self.uids["pretrain"] & self.uids["eval"]), (
            "Pretrain and eval datasets must not share any units"
        )

        self.finetune_ckpts = resolve_finetune_ckpts(
            seed=self.seed,
            ckpt_dir=self.ckpt_dir,
            pretrain_ckpt=self.pretrain_ckpt,
            filters=self.filters,
            csv_path=self.csv_path,
        )
        logger.info(f"seed {self.seed}: {len(self.finetune_ckpts)} finetuning checkpoints")

    @abstractmethod
    def read(
        self, state_dict: dict, dataset: WholeSessionSpikeDataset, uids: set[str]
    ) -> tuple[torch.Tensor, np.ndarray]:
        """Embeddings for ``uids`` out of one checkpoint's weights, and the uids they match."""

    def _read_ckpt(
        self, path, dataset, uids, expect_seed: int | None = None
    ) -> tuple[torch.Tensor, np.ndarray]:
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        if expect_seed is not None:
            # two seeds of one recording match in shape, so nothing else catches a swap
            trained_with = ckpt["cfg"].get("seed")
            assert trained_with in (expect_seed, None), (
                f"{path} was trained with seed {trained_with}, not {expect_seed}: whatever "
                "listed it disagrees with the checkpoint"
            )
        return self.read(ckpt["model_state_dict"], dataset, uids)

    def encode(self, regime: BenchmarkRegime) -> tuple[torch.Tensor, np.ndarray]:
        if regime == "pretrain":
            return self._read_ckpt(self.pretrain_ckpt, self.datasets[regime], self.uids[regime])

        dataset = self.datasets[regime]
        all_embs, all_uids = [], []
        pbar = tqdm(self.finetune_ckpts, desc=f"[emb-{regime}]")
        for entry in pbar:
            rec_uids = set(dataset.get_recording(entry.recording_id).units.id.astype(str))
            embs, uids = self._read_ckpt(entry.path, dataset, rec_uids, expect_seed=self.seed)
            all_embs.append(embs)
            all_uids.append(uids)
            pbar.set_postfix(units=sum(len(u) for u in all_uids))

        return torch.cat(all_embs), np.concatenate(all_uids)
