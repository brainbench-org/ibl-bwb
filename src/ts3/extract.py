"""Stage 2: a frozen checkpoint in, one unit-embeddings file out.

One extractor object encodes both regimes, because the two halves of the file are only
comparable if the same thing produced them. A probe fit on one encoder's features and applied
to another's fails silently: it scores like a weak model rather than a broken one.

This step reads pretrain units too, and that is not a leak, because it tunes nothing. Fitting
the probe on pretrain units and their labels is the protocol. What matters is the other
direction: the eval regime is named here, and nowhere under ``pretrain/``, which the
extractors are parameterised by regime to allow.

Which extractor runs is a config choice (``extractor=``), resolved the same way ``trainer=``
is in TS1 and TS2. What each one can do with the eval regime is not a choice: see
``ts3/models/inductive/`` and ``ts3/models/transductive/``.
"""

import hydra
import torch
from omegaconf import DictConfig

from core.launch import run
from core.utils.logger import Logger
from core.utils.unit_embeddings import save_unit_embeddings
from core.utils.util import expand_path


@hydra.main(version_base="1.2", config_path="./configs", config_name="extract.yaml")
def main(cfg: DictConfig):
    assert cfg.data_root, (
        "data_root is unset: set BWB_DATA_ROOT, its per-build override, or pass data_root="
    )
    logger = Logger(rank=0)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    extractor = hydra.utils.instantiate(cfg.extractor)
    extractor.setup(expand_path(cfg.data_root), device, logger)

    train_embs, train_uids = extractor.encode("pretrain")
    eval_embs, eval_uids = extractor.encode("eval")

    path = save_unit_embeddings(
        expand_path(cfg.embs_dir) / f"{cfg.name or extractor.name}.pt",
        train_embs=train_embs,
        train_uids=train_uids,
        eval_embs=eval_embs,
        eval_uids=eval_uids,
        run_seed=extractor.run_seed,
    )
    logger.info(f"Embeddings saved at {path}")


if __name__ == "__main__":
    run(main)
