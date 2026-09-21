from typing import Literal, get_args

import torch
from torch import nn

from .neds import ALL_MODALITIES

NEDSMaskType = Literal[
    "neural_masking",
    "behavior_masking",
    "within_neural",
    "within_behavior",
    "cross_modality",
]


# Mask is a classic NDT mask modeling (BERT style)
class NEDSMasker(nn.Module):
    def __init__(
        self,
        mask_ratio,
        modalities=ALL_MODALITIES,
        mask_types=None,
    ):
        super().__init__()
        self.mask_ratio = mask_ratio
        self.modalities = modalities
        self.mask_types = mask_types if mask_types is not None else list(get_args(NEDSMaskType))

    def _random_mask(self, shape, device, generator=None):
        return torch.rand(shape, device=device, generator=generator) < self.mask_ratio

    def _generate_mask_map(self, shape, device, generator=None):
        mask_map = {}

        all_masked = torch.ones(shape, dtype=torch.bool, device=device)
        not_masked = torch.zeros(shape, dtype=torch.bool, device=device)
        # one shared draw, so a cross-modality sample masks the same timesteps everywhere
        cross_modality_random_masked = self._random_mask(shape, device, generator)

        for modality in self.modalities:
            if modality == "spikes":
                mask_map[modality] = {
                    "neural_masking": all_masked,
                    "behavior_masking": not_masked,
                    "within_neural": self._random_mask(shape, device, generator),
                    "within_behavior": not_masked,
                    "cross_modality": cross_modality_random_masked,
                }
            else:
                mask_map[modality] = {
                    "neural_masking": not_masked,
                    "behavior_masking": all_masked,
                    "within_neural": not_masked,
                    "within_behavior": self._random_mask(shape, device, generator),
                    "cross_modality": cross_modality_random_masked,
                }

        return mask_map

    def forward(self, input, generator=None):
        """``generator``: draw from it instead of the global RNG, to hold the masks
        fixed across epochs (the validation pass seeds one per epoch).
        """
        B, T, *_ = input.size()
        shape = (B, T, 1)
        device = input.device

        modality_mask = {}

        scheme_idx = torch.randint(
            len(self.mask_types), (B,), device=device, generator=generator
        )  # (B,)
        mask_map = self._generate_mask_map(shape, device, generator)  # {mod: (B, T, 1)}
        rows = torch.arange(B, device=device)

        for modality in self.modalities:
            # (S, B, T, 1) -> (B, T, 1), keeping each sample's own scheme and draw
            per_scheme = torch.stack([mask_map[modality][scheme] for scheme in self.mask_types])

            modality_mask[modality] = per_scheme[scheme_idx, rows]

        return modality_mask
