"""The unit-embeddings file: one writer, one reader.

Written by any model that learns a per-unit representation (NuCLR and NEMO under
``src/pretrain/``, the ISI feature and the transductive extractor under ``src/ts3/``) and
read back by the TS3 probes. Keeping both ends here is what stops the key names from
drifting apart. Joining the result to a unit table is a suite's business, not this
module's: see :func:`ts3.embeddings.load_embs_and_md`.
"""

from pathlib import Path

import numpy as np
import torch

from core.utils.logger import get_cli_logger
from core.utils.util import expand_path

logger = get_cli_logger()


def save_unit_embeddings(
    path,
    train_embs,
    train_uids,
    eval_embs,
    eval_uids,
    run_id: str | None = None,
    run_seed: int | None = None,
) -> Path:
    """Write the embeddings file :func:`load_unit_embeddings` reads, and return where.

    ``path`` is either the file to write or a directory to write ``<run_id>.pt`` into;
    anything without a ``.pt`` suffix is taken as a directory, and then ``run_id`` is
    required. Naming by run id keeps two runs of the same model from overwriting each
    other and is the key that joins the file back to its W&B run. Tensors are moved to
    the host, so the file loads without a GPU. Call it on rank 0 only.

    ``run_seed``, the seed of the run behind the embeddings, is what a submission is filed
    under; None when nothing trained them.
    """
    path = expand_path(path)
    if path.suffix != ".pt":
        if run_id is None:
            raise ValueError(f"{path} names a directory, so run_id is needed to name the file")
        path = path / f"{run_id}.pt"
    path.parent.mkdir(parents=True, exist_ok=True)

    torch.save(
        {
            "train_embs": train_embs.detach().cpu(),
            "train_uids": np.asarray(train_uids),
            "eval_embs": eval_embs.detach().cpu(),
            "eval_uids": np.asarray(eval_uids),
            "run_seed": run_seed,
        },
        path,
    )
    return path


def load_unit_embeddings(emb_path):
    """Read the arrays back as numpy, dropping units whose embedding is not finite.

    The run seed comes back last, None for a file that recorded none.
    """
    loaded = torch.load(emb_path, map_location="cpu", weights_only=False)
    train_embs = loaded["train_embs"].detach().numpy()
    train_uids = loaded["train_uids"]
    eval_embs = loaded["eval_embs"].detach().numpy()
    eval_uids = loaded["eval_uids"]

    mask = np.all(np.isfinite(train_embs), axis=-1)
    if not mask.all():
        logger.info(f"{(~mask).sum()} train_embs are not finite")
    train_embs, train_uids = train_embs[mask], train_uids[mask]

    mask = np.all(np.isfinite(eval_embs), axis=-1)
    if not mask.all():
        logger.info(f"{(~mask).sum()} eval_embs are not finite")
    eval_embs, eval_uids = eval_embs[mask], eval_uids[mask]

    return train_embs, train_uids, eval_embs, eval_uids, loaded.get("run_seed")
