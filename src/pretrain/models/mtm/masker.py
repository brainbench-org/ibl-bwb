import random
from typing import Literal

import torch
from torch import nn

MtMMaskType = Literal["neuron", "causal", "inter_region", "intra_region"]


class MtMMasker(nn.Module):
    r"""Module for masking spikes. Masking modes:

    - ``neuron``: mask all time bins of a random set of neurons.
    - ``causal``: mask a fixed set of future time steps.
    - ``inter_region``: mask all neurons of one region; predictions are for that masked region.
    - ``intra_region``: randomly mask a fraction of neurons within a target region; predict those
      masked neurons from the rest of the target region only (all neurons outside the
      target region are zeroed out in the input).

    CONFIG:

    - ``mask_ratio``: fraction of neurons to mask (used by neuron and intra_region modes)
    - ``causal_ratio``: fraction of trailing time bins to mask (deterministic; the last
      ``floor(causal_ratio * T)`` bins are masked, with a floor of 1 bin).
    """

    def __init__(
        self,
        mask_ratio,
        mask_types,
        causal_ratio: float = 0.1,
    ):
        super().__init__()

        self.mask_ratio = mask_ratio
        self.mask_types = mask_types
        self.causal_ratio = causal_ratio

    def forward(
        self,
        spikes,
        regions,
        mask_mode: MtMMaskType | None = None,
        generator: torch.Generator | None = None,
    ):
        if mask_mode is None:
            mask_mode = random.choice(self.mask_types)

        B, T, N = spikes.shape

        if mask_mode in ["inter_region", "intra_region"]:
            assert regions is not None, "Can't mask region without brain region information"
            unique_regions = torch.unique(regions)

        if mask_mode == "causal":
            mask_length = max(1, int(self.causal_ratio * T))
            cols_idx = torch.arange(T, device=spikes.device)  # (T,)
            is_mask = (cols_idx >= (T - mask_length)).unsqueeze(0).expand(B, -1)

        elif mask_mode == "neuron":
            is_mask = (
                torch.rand((B, N), device=spikes.device, generator=generator) < self.mask_ratio
            )

        elif mask_mode == "inter_region":
            idx = torch.randint(
                len(unique_regions), (1,), device=spikes.device, generator=generator
            ).item()
            masked_region = unique_regions[idx]

            is_mask = torch.where(regions == masked_region, True, False)

        elif mask_mode == "intra_region":
            idx = torch.randint(
                len(unique_regions), (1,), device=spikes.device, generator=generator
            ).item()
            target_region = unique_regions[idx]

            is_mask = (
                torch.rand((B, N), device=spikes.device, generator=generator) < self.mask_ratio
            )
            is_mask = torch.where(regions == target_region, is_mask, False)

        if mask_mode == "causal":
            is_mask = is_mask.unsqueeze(-1).expand(-1, -1, N)  # (B, T, N)

        if mask_mode in ["neuron", "intra_region", "inter_region"]:
            is_mask = is_mask.unsqueeze(1).expand(-1, T, -1)  # (B, T, N)

        spikes_zeroed = spikes.clone()
        spikes_zeroed[is_mask] = 0.0
        if mask_mode == "intra_region":
            _regions = regions.unsqueeze(1).expand(-1, T, -1)  # (B, T, N)
            spikes_zeroed[_regions != target_region] = 0.0

        return spikes_zeroed, is_mask, mask_mode
