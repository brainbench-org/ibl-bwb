import torch
import torch.nn.functional as F
from torch import nn


class NDTStitchMasker(nn.Module):
    r"""Temporal masking for spikes (BERT-style).

    Randomly masks time bins across all neurons. Each call masks either independent
    bins or contiguous time spans, drawn per call.

    Args:
        mask_ratio: Fraction of time bins to mask.
        max_block_size: Maximum block size for contiguous masking, 1 for independent
            per-bin masking.
        block_mask_prob: Probability of masking a contiguous block rather than random
            bins, drawn per call and only read when ``max_block_size`` is above 1.
    """

    def __init__(
        self,
        mask_ratio,
        max_block_size=1,
        block_mask_prob=0.5,
    ):
        super().__init__()

        self.mask_ratio = mask_ratio
        self.max_block_size = max_block_size
        self.block_mask_prob = block_mask_prob

    def generate_block_mask(self, shape, device):
        """
        Generates a mask where points are grouped into contiguous blocks.

        Args:
            shape: Tuple of (Batch size, Time steps)
            device: Torch device
        """
        B, T = shape

        # We reduce the initial probability to keep the total masked area consistent.
        block_size = torch.randint(1, self.max_block_size + 1, (1,)).item()
        block_mask_ratio = self.mask_ratio / block_size

        # Seed the mask with sparse "seed" points
        mask = torch.rand((B, T), device=device) < block_mask_ratio

        # Expand seeds into blocks
        # MaxPool1d acts as a 'dilator', if any 'True' is in the window,
        # the whole window becomes 'True'.
        mask = mask.float().unsqueeze(1)  # (N, 1, T)

        mask = F.max_pool1d(mask, kernel_size=block_size, stride=1, padding=block_size // 2)

        # Correct for the 'one-off' error that occurs with even kernels
        # during symmetric padding
        if block_size % 2 == 0:
            mask = mask[..., :-1]

        mask = mask.squeeze(1)  # (N, T)

        return mask.bool()

    def forward(self, spikes):
        B, T, N = spikes.size()
        masked_spikes = spikes.clone()

        should_generate_block_mask = (
            self.max_block_size > 1 and torch.rand(1).item() < self.block_mask_prob
        )

        if should_generate_block_mask:
            is_mask = self.generate_block_mask((B, T), device=spikes.device)
        else:
            is_mask = torch.rand((B, T), device=spikes.device) < self.mask_ratio

        is_mask = is_mask.unsqueeze(-1).expand(-1, -1, N)  # (B, T) -> (B, T, N)
        masked_spikes[is_mask] = 0

        return masked_spikes, is_mask
