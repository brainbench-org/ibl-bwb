import torch
import torch.nn as nn


class NDT2Masker(nn.Module):
    """ShuffleInfill-style MAE masker for NDT2.

    A single random shuffle is shared across the whole batch (same as
    ShuffleInfill).  The first ``encoder_frac`` positions go to the encoder;
    the rest become decoder queries.  All tensors keep their full padded
    length, no per-sample packing needed.  The loss uses ``query_mask``
    to ignore padding and spatially-invalid tokens.
    """

    def __init__(self, mask_ratio: float):
        super().__init__()
        self.mask_ratio = mask_ratio

    def forward(self, inputs):
        B, seq_len = inputs.size(0), inputs.size(1)

        encoder_frac = round((1 - self.mask_ratio) * seq_len)

        shuffle = torch.randperm(seq_len, device=inputs.device)
        mask = torch.zeros(B, seq_len, dtype=torch.bool, device=inputs.device)
        mask[:, shuffle[encoder_frac:]] = True

        return mask
