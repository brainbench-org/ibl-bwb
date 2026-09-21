"""Finding the per-session finetuning checkpoints the transductive regime reads.

``ckpt_dir`` is scanned for every ``best.pt`` whose config names one recording, a seed, and
the pretrain checkpoint's model. One per (seed, recording) is required: two of them raise
rather than pick. ``filters`` narrows on any config field, ``csv_path`` lists the runs by
hand instead.
"""

import ast
import os
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import torch
from omegaconf import OmegaConf

from core.utils.logger import get_cli_logger

logger = get_cli_logger()

CKPT_NAME = "best.pt"


@dataclass(frozen=True)
class FinetuneCkpt:
    recording_id: str
    path: str


def read_ckpt_cfg(path) -> dict:
    """The config a checkpoint was trained with, without paging in its weights."""
    try:
        ckpt = torch.load(path, map_location="cpu", weights_only=False, mmap=True)
    except Exception:  # a checkpoint saved in the legacy format cannot be mapped
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
    cfg = ckpt.get("cfg")
    if cfg is None:
        return {}
    try:
        return dict(cfg)
    except Exception:
        # The scan reads every best.pt under ckpt_dir before it can filter any out, so an
        # unrelated run's config must not be able to end it. `dict()` resolves as it goes,
        # and a suite that leaves a mandatory value unset (ts1 and ts2 both ship
        # ``task: ???``) raises here for a checkpoint this scan was going to discard
        # anyway. Read it unresolved instead; the fields matched on below are all literals.
        try:
            return OmegaConf.to_container(cfg, resolve=False, throw_on_missing=False)
        except Exception:
            return {}


def _get(cfg, dotted: str):
    """Look up a dotted config key, so a filter can name ``model.dim`` as well as ``task``."""
    node = cfg
    for part in dotted.split("."):
        try:
            node = node[part]
        except (KeyError, TypeError, IndexError):
            return None
    return node


def _same_model(a, b) -> bool:
    """Whether two ``model._target_`` strings name the same class.

    Compared on the class name, not the dotted path: a pretrain checkpoint predating a
    module move records the old path (``pretrain.models.POYOPlus``) while a calibration
    run records the new one (``pretrain.models.poyo_plus.POYOPlus``), and a full-path
    comparison silently matches nothing. The class name still separates POYOPlus from
    POSSM, MtM and NDTStitch, which is all this filter is for.
    """
    if a is None or b is None:
        return a == b
    return str(a).rsplit(".", 1)[-1] == str(b).rsplit(".", 1)[-1]


def _unwrap_recording_id(val):
    """Handle case where recording_ids is serialized as '["recording_id"]'."""
    if isinstance(val, str) and val.startswith("["):
        parsed = ast.literal_eval(val)
        assert len(parsed) == 1, f"expected singleton recording_ids, got {parsed}"
        return parsed[0]
    return val


def _from_ckpt_dir(
    ckpt_dir: str, model_target: str | None, filters: dict | None
) -> dict[int, list[FinetuneCkpt]]:
    root = Path(ckpt_dir)
    if not root.is_dir():
        raise FileNotFoundError(f"ckpt_dir {root} is not a directory")
    paths = sorted(root.rglob(CKPT_NAME))
    out: dict[int, list[FinetuneCkpt]] = defaultdict(list)
    for path in paths:
        cfg = read_ckpt_cfg(path)
        rids, seed = _get(cfg, "recording_ids"), _get(cfg, "seed")
        # a pretraining run holds every recording, a finetuning run holds one
        if seed is None or not rids or len(rids) != 1:
            continue
        if model_target is not None and not _same_model(_get(cfg, "model._target_"), model_target):
            continue
        if filters and any(_get(cfg, key) != value for key, value in filters.items()):
            continue
        out[int(seed)].append(FinetuneCkpt(str(rids[0]), str(path)))

    kept = sum(len(v) for v in out.values())
    scope = f"model {model_target}" if model_target else "no model filter"
    logger.info(f"{len(paths)} checkpoints under {ckpt_dir}, {kept} finetuning runs ({scope})")
    return dict(out)


def _from_csv(csv_path: str, ckpt_dir: str) -> dict[int, list[FinetuneCkpt]]:
    df = pd.read_csv(csv_path)
    required = ["ID", "recording_ids", "seed"]
    assert all(col in df.columns for col in required), f"csv must contain columns: {required}"

    out: dict[int, list[FinetuneCkpt]] = defaultdict(list)
    use_csv_path = "ckpt_path" in df.columns
    if use_csv_path:
        logger.warning(
            "csv contains ckpt_path column, using it instead of inferring from ID and ckpt_dir"
        )
    else:
        logger.warning(f"Inferring ckpt path from ID and ckpt_dir (choosing {CKPT_NAME} for all)")

    for _, row in df.iterrows():
        seed = int(row["seed"])
        rec_id = _unwrap_recording_id(row["recording_ids"])

        if use_csv_path and pd.notna(row["ckpt_path"]):
            path = str(row["ckpt_path"])
        else:
            path = str(Path(ckpt_dir) / row["ID"] / CKPT_NAME)

        out[seed].append(FinetuneCkpt(rec_id, path))

    return dict(out)


def _check_one_per_recording(index: dict[int, list[FinetuneCkpt]]) -> None:
    """One run per (seed, recording): two of them is an error, not a choice."""
    for seed, entries in sorted(index.items()):
        repeated = {r: n for r, n in Counter(e.recording_id for e in entries).items() if n > 1}
        if repeated:
            lines = [f"seed {seed} has more than one checkpoint for a recording:"]
            for rid, n in sorted(repeated.items()):
                paths = sorted(e.path for e in entries if e.recording_id == rid)
                lines.append(f"  {rid}: {n} checkpoints, e.g. {paths[:3]}")
            lines.append("Prune the directory, narrow filters, or list the runs in a csv.")
            raise ValueError("\n".join(lines))


def _check_even_coverage(index: dict[int, list[FinetuneCkpt]]) -> None:
    """Every seed should cover the same recordings, or the seeds are not comparable."""
    per_seed = {seed: {e.recording_id for e in entries} for seed, entries in index.items()}
    if len({frozenset(rids) for rids in per_seed.values()}) > 1:
        lines = ["seeds cover different recordings:"]
        lines += [
            f"  seed={seed}: {len(rids)} recordings" for seed, rids in sorted(per_seed.items())
        ]
        raise ValueError("\n".join(lines))


def resolve_finetune_ckpts(
    seed: int,
    ckpt_dir: str,
    pretrain_ckpt: str | Path | None = None,
    filters: dict | None = None,
    csv_path: str | None = None,
) -> list[FinetuneCkpt]:
    """The checkpoints for one seed, after checking the set they came from is coherent."""
    if csv_path is not None:
        index = _from_csv(csv_path, ckpt_dir)
    else:
        # the model comes from the pretrain checkpoint, so one dir may hold several
        target = _get(read_ckpt_cfg(pretrain_ckpt), "model._target_") if pretrain_ckpt else None
        index = _from_ckpt_dir(ckpt_dir, target, filters)

    _check_one_per_recording(index)
    _check_even_coverage(index)
    if seed not in index:
        raise ValueError(f"No finetuning runs for seed {seed}; found {sorted(index)}")

    entries = index[seed]
    missing = [e.path for e in entries if not os.path.exists(e.path)]
    if missing:
        raise FileNotFoundError(
            f"{len(missing)} of {len(entries)} finetuning checkpoints for seed {seed} do not "
            f"exist, e.g. {missing[:5]}"
        )
    return entries
