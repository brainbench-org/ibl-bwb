import logging

import torch
import torch.nn.functional as F

logger = logging.getLogger(__name__)


def rankme(x: torch.Tensor, normalize: bool = True) -> float:
    """Effective rank of an ``(entities, dim)`` embedding: the entropy of its spectrum."""
    assert x.ndim == 2  # (neurons, dim)

    invalid_emb = (~x.isfinite()).sum(dim=1) > 0
    x = x[~invalid_emb, :]

    if normalize:
        x = F.normalize(x, dim=-1)

    try:
        cov = torch.cov(x.T).float()
        eigvals = torch.linalg.eigvalsh(cov)
        eigvals = eigvals / eigvals.sum()
        eigvals = eigvals[eigvals > 0]
        rankme = torch.exp(-(eigvals * torch.log(eigvals)).sum())
    except Exception:
        logger.error("Error computing rankme", exc_info=True)
        rankme = torch.nan
    return rankme.item()
