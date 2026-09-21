"""Joining a unit-embeddings file to the TS3 unit table.

The file format itself is generic and lives in ``core.utils.unit_embeddings``. What is TS3
here is which units count: the join drops anything the protocol does not score, and insists
the eval side is complete.
"""

from pathlib import Path

import numpy as np
import pandas as pd

from core.utils.unit_embeddings import load_unit_embeddings
from ibl_bwb_eval.tasks import TS3Task
from ts3.protocol import get_md_from_dataset


def load_embs_and_md(
    data_root: Path, emb_path: Path, task: TS3Task
) -> tuple[np.ndarray, pd.DataFrame, np.ndarray, pd.DataFrame]:
    """Both halves of an embeddings file, each joined to the TS3 unit table.

    The pretrain part may be short, since an extractor may omit units it cannot represent.
    The eval part may not: a missing unit would shrink what a submission covers in silence.
    The file's run seed rides along in ``eval_md.attrs``, beside the build's provenance.
    """
    train_embs, train_uids, eval_embs, eval_uids, run_seed = load_unit_embeddings(emb_path)

    # Train
    train_md = get_md_from_dataset(data_root, "pretrain", task)

    mask = pd.Index(train_uids).isin(train_md.index)  # faster than np.isin
    train_embs, train_uids = train_embs[mask], train_uids[mask]
    train_md = train_md.loc[train_uids]

    # Eval
    eval_md = get_md_from_dataset(data_root, "eval", task)
    mask = pd.Index(eval_uids).isin(eval_md.index)  # faster than np.isin
    eval_embs, eval_uids = eval_embs[mask], eval_uids[mask]
    if len(eval_uids) != len(eval_md):
        raise ValueError(
            f"Some eval embeddings are missing. Got {len(eval_uids)}, expected {len(eval_md)}"
        )
    eval_md = eval_md.loc[eval_uids]
    eval_md.attrs["run_seed"] = run_seed

    return train_embs, train_md, eval_embs, eval_md
