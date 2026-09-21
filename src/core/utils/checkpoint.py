import os
from collections import Counter
from collections.abc import Sequence
from pathlib import Path

import torch


def save_ckpt(filepath: Path, **kwargs) -> Path:
    checkpoint = {
        "git_hash": os.popen("git rev-parse HEAD").read().strip(),
    }

    for k, v in kwargs.items():
        if isinstance(v, torch.nn.parallel.DistributedDataParallel):
            checkpoint[f"{k}_state_dict"] = v.module.state_dict()
        elif hasattr(v, "state_dict"):
            checkpoint[f"{k}_state_dict"] = v.state_dict()
        else:
            checkpoint[k] = v

    filepath.parent.mkdir(exist_ok=True, parents=True)
    torch.save(checkpoint, filepath)
    return filepath


def _summarize(keys: list[str], max_groups: int = 6) -> str:
    groups = Counter(k.rsplit(".", 1)[0] for k in keys)
    txt = ", ".join(f"{name} ({n})" for name, n in groups.most_common(max_groups))
    if len(groups) > max_groups:
        txt += f", +{len(groups) - max_groups} more"
    return txt


def log_incompatible_keys(result, logger, *, ignore: Sequence[str] = ()) -> None:
    """Report what a ``strict=False`` load left untouched.

    Missing tensors keep their initialization, so they are only safe when the caller
    means to refit them: pass those key substrings in ``ignore`` to log them as such.
    Shape mismatches are not covered here, ``load_state_dict`` raises on those either way.
    """
    warn = getattr(logger, "warning", None) or logger.warn

    def split(keys):
        known, rest = [], []
        for k in keys:
            (known if any(s in k for s in ignore) else rest).append(k)
        return known, rest

    known_missing, missing = split(result.missing_keys)
    known_unused, unexpected = split(result.unexpected_keys)
    known = known_missing + known_unused

    if known:
        logger.info(f"Skipped by design ({len(known)}): {_summarize(known)}")
    if missing:
        warn(f"Missing from checkpoint, kept at init ({len(missing)}): {_summarize(missing)}")
    if unexpected:
        warn(f"In checkpoint but unused by the model ({len(unexpected)}): {_summarize(unexpected)}")
    if not missing and not unexpected:
        logger.info("Checkpoint keys matched the model")
