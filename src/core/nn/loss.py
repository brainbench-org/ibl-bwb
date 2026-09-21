"""Losses shared by more than one model's objective."""

import torch
import torch.nn.functional as F
from torch.distributed.nn import all_gather

from core.utils.distributed import is_distributed

__api_ref__ = {
    "description": None,
    "sections": [{"title": None, "autosummary": ["WeightedMSELoss", "CLIPLoss"]}],
}


class WeightedMSELoss(torch.nn.Module):
    """Mean squared error under per-element weights, normalized by their sum."""

    def forward(
        self,
        input: torch.Tensor,
        target: torch.Tensor,
        weights: torch.Tensor,
    ) -> torch.Tensor:
        r"""Compute mean squared error loss.

        Args:
            input: :math:`(N, D)` float, model predictions.
            target: :math:`(N, D)` float, ground truth targets.
            weights: :math:`(N, D)` float, per-element loss weights.
        """
        if input.ndim != 2:
            raise ValueError("Input must have 2 dimensions")
        if target.ndim != 2:
            raise ValueError("Target must have 2 dimensions")
        if weights.ndim != 2:
            raise ValueError("Weights must have 2 dimensions")
        if input.shape[0] != weights.shape[0]:
            raise ValueError("Input and weights must have the same batch size")
        if weights.sum() < 1e-9:
            raise ValueError("Expected non-degenerate weights (all weights are zero)")

        loss_noreduce = F.mse_loss(input, target, reduction="none")
        return (weights * loss_noreduce).sum() / weights.sum()


class CLIPLoss(torch.nn.Module):
    r"""Symmetric InfoNCE loss over two paired views (CLIP, Eq. 1 of the NEMO paper).

    Args:
        temperature: softmax temperature applied to the cosine similarities.
        gather_distributed: take the negatives from the global batch by pooling both
            views across ranks, so the objective does not depend on the GPU count.
            Every rank must hold the same number of samples.
    """

    def __init__(self, temperature: float = 0.5, gather_distributed: bool = True):
        super().__init__()
        self.temperature = temperature
        self.gather_distributed = gather_distributed

    def forward(self, z_a: torch.Tensor, z_b: torch.Tensor) -> torch.Tensor:
        r"""Compute the loss for row-wise paired embeddings.

        Args:
            z_a: :math:`(N, D)` float, embeddings of one view.
            z_b: :math:`(N, D)` float, embeddings of the other view.
        """
        z_a = F.normalize(z_a, dim=-1)
        z_b = F.normalize(z_b, dim=-1)
        if self.gather_distributed and is_distributed():
            z_a = torch.cat(all_gather(z_a), dim=0)
            z_b = torch.cat(all_gather(z_b), dim=0)

        sim = z_a @ z_b.T / self.temperature
        labels = torch.arange(len(sim), device=sim.device)
        return (F.cross_entropy(sim, labels) + F.cross_entropy(sim.T, labels)) / 2
