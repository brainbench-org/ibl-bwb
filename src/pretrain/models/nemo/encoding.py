"""NEMO's encoding contract: how a unit becomes a vector, in one place.

The waveform peak-normalisation, the ACG scale factor and the ``[acg, wf]`` concat order are
the model's, so every caller reads them from here: the online monitor during training, and
``ts3.models.inductive.nemo`` afterwards. The dataset is a parameter throughout, so this
module names no regime and no QC policy: the population is the caller's choice.
"""

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from core.utils.logger import Logger

from .cache import load_cache, update_cache

ACG_MULTI_FACTOR = 10.0


def preprocess(
    wf: torch.Tensor, acg: torch.Tensor, device: torch.device
) -> tuple[torch.Tensor, torch.Tensor]:
    """Peak-normalise each waveform and put the ACGs on the scale the encoder saw."""
    wf = wf / (wf.abs().amax(dim=-1, keepdim=True) + 1e-8)
    return wf.float().to(device), (acg * ACG_MULTI_FACTOR).float().to(device)


@torch.inference_mode()
def encode(
    model,
    waveforms: torch.Tensor,
    acgs: torch.Tensor,
    batch_size: int,
    device: torch.device,
    dtype: torch.dtype,
    logger: Logger,
) -> torch.Tensor:
    """``[acg_rep, wf_rep]`` per unit, in NEMO's concat order."""
    model = getattr(model, "module", model)  # DDP does not proxy representation()
    model.eval()
    loader = DataLoader(TensorDataset(waveforms, acgs), batch_size=batch_size, shuffle=False)

    reps = []
    for wf_batch, acg_batch in logger.get_pbar(loader, prefix="embed"):
        wf_batch, acg_batch = preprocess(wf_batch, acg_batch, device)
        with torch.autocast(device_type=device.type, dtype=dtype):
            wf_rep, acg_rep = model.representation(wf_batch, acg_batch.unsqueeze(1))
        reps.append(torch.cat([acg_rep, wf_rep], dim=1).cpu().float())

    return torch.cat(reps)


def load_units(cache_path, dataset) -> tuple[torch.Tensor, torch.Tensor, np.ndarray]:
    """Waveforms, ACGs and uids for a dataset's units, building the cache on first use."""
    update_cache(Path(cache_path), dataset)
    data = load_cache(Path(cache_path), dataset)

    eids = set(dataset.get_session_ids())
    mask = np.array([eid in eids for eid in data["eids"]])
    return (
        torch.from_numpy(data["waveforms"][mask]).float(),
        torch.from_numpy(data["acgs"][mask]).float(),
        data["uuids"][mask],
    )
