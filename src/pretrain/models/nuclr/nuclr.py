import numpy as np
import torch
import torch.nn as nn
from torch import Tensor
from torch_brain.batching import chain
from torch_brain.nn import RotaryTimeEmbedding

from core.utils.util import is_divisible

from .attention import SelfAttention


class NuCLR(nn.Module):
    """Contrastive unit encoder over population context :cite:`nuclr`.

    Reference implementation: `nerdslab/nuclr <https://github.com/nerdslab/nuclr>`_, with the
    deviations listed in ``src/pretrain/models/nuclr/README.md``.
    """

    def __init__(
        self,
        ctx_duration: float,
        latent_step: float,
        bin_size: float,
        dim: int,
        self_heads: int,
        dim_head: int,
        lin_dropout: float,
        t_layers: int,
        st_layers: int,
        rot_ratio: float = 0.5,
        rotate_value: bool = True,
        attn_biases: bool = True,
    ):
        super().__init__()

        assert is_divisible(latent_step, bin_size)
        assert is_divisible(ctx_duration, latent_step)
        self.ctx_duration = ctx_duration
        self.latent_step = latent_step
        self.bin_size = bin_size
        self.emb_dim = self.dim = dim

        # derived params
        self.bins_per_latent = int(latent_step / bin_size)  # num of bins per latent
        self.num_latents = int(np.ceil(ctx_duration / latent_step))

        t_min, t_max = 1.0, 8.0 * self.num_latents
        self.rotary_emb = RotaryTimeEmbedding(
            head_dim=dim_head,
            rotate_dim=int(dim_head * rot_ratio),
            t_min=t_min,
            t_max=t_max,
        )

        self.read_in = nn.Linear(self.bins_per_latent, dim)

        self.t_blocks: list[list[nn.Module]] = nn.ModuleList(
            [
                nn.ModuleList(
                    [
                        SelfAttention(
                            dim=dim,
                            heads=self_heads,
                            dim_head=dim_head,
                            rotate_value=rotate_value,
                            to_qkv_bias=attn_biases,
                        ),
                        FFN(dim=dim, mult=4, dropout=lin_dropout, pre_norm=True),
                    ]
                )
                for _ in range(t_layers)
            ]
        )  # ty:ignore[invalid-assignment]

        self.st_blocks: list[list[nn.Module]] = nn.ModuleList(
            [
                nn.ModuleList(
                    [
                        SelfAttention(
                            dim=dim,
                            heads=self_heads,
                            dim_head=dim_head,
                            to_qkv_bias=attn_biases,
                        ),
                        FFN(dim=dim, mult=4, dropout=lin_dropout, pre_norm=True),
                        SelfAttention(
                            dim=dim,
                            heads=self_heads,
                            dim_head=dim_head,
                            rotate_value=rotate_value,
                            to_qkv_bias=attn_biases,
                        ),
                        FFN(dim=dim, mult=4, dropout=lin_dropout, pre_norm=True),
                    ]
                )
                for _ in range(st_layers)
            ]
        )  # ty:ignore[invalid-assignment]

        self.dp = nn.Dropout(lin_dropout)

    def forward(self, bins: Tensor, spax_seqlen: Tensor) -> Tensor:
        r"""
        Args:
            bins: Chained tensor of binned spikes ``(*num_neurons, num_bins)``
            spax_seqlen: seqlen to constrain spatial attention
        """
        device = bins.device
        num_units = int(spax_seqlen.sum())
        spax_seqlen = spax_seqlen.repeat(self.num_latents)

        # (U, L*B) -> (U, L, B)

        x = bins.reshape(bins.shape[0], self.num_latents, self.bins_per_latent)

        x = self.read_in(x.float())
        t = torch.arange(self.num_latents, device=device).repeat(num_units, 1).float()
        x_rot = self.rotary_emb(t)

        # Temporal attention layers
        for t_attn, t_ffn in self.t_blocks:
            x = x + self.dp(t_attn(x, rotary=x_rot))
            x = x + self.dp(t_ffn(x))

        # Spatio-temporal attention layers
        for s_attn, s_ffn, t_attn, t_ffn in self.st_blocks:
            # Spatial
            x = x.permute(1, 0, 2).reshape(-1, x.shape[2])  # (U, L, D) -> (L*U, D)
            x = x + self.dp(s_attn(x, seqlen=spax_seqlen))
            x = x + self.dp(s_ffn(x))
            x = x.reshape(self.num_latents, num_units, -1).permute(1, 0, 2)  # (L*U, D) -> (U, L, D)

            # Temporal
            x = x + self.dp(t_attn(x, rotary=x_rot))
            x = x + self.dp(t_ffn(x))

        y = x.mean(1)
        return y

    def input_fn(self, data) -> dict | None:
        if len(data.units) < 2:
            return None  # cannot work with a single unit

        spike_times = torch.tensor(data.spikes.timestamps, dtype=torch.float32)
        spike_units = torch.tensor(data.spikes.unit_index, dtype=torch.long)
        num_units = len(data.units.id)

        # bin spikes
        num_bins = int(self.ctx_duration / self.bin_size)
        rate = 1 / self.bin_size
        bins = torch.zeros((num_units, num_bins + 1), dtype=torch.int16)
        bins.index_put_(
            indices=(spike_units, torch.floor(spike_times * rate).long()),
            values=torch.ones(len(spike_times), dtype=torch.int16),
            accumulate=True,
        )
        bins = bins[:, : self.num_latents * self.bins_per_latent]

        pids = data.units.probe_id.astype(str)
        _, probe_counts = np.unique(pids, return_counts=True)
        probe_boundary_idx = np.where(pids[1:] != pids[:-1])[0]
        assert len(probe_boundary_idx) <= 1
        assert len(probe_boundary_idx) == len(probe_counts) - 1

        return {
            "model_inputs": {
                "bins": chain(bins),
                "spax_seqlen": chain(probe_counts),
            },
            "unit_seqlen": num_units,
            "unit_ids": data.units.id.astype(str).tolist(),
            "probe_ids": data.units.probe_id.astype(str).tolist(),
        }

    @staticmethod
    def collate(data: list[dict | None], two_view: bool = False):
        import itertools

        from torch_brain.batching import collate as tb_collate

        X = [x for x in data if x is not None]

        if two_view:
            # flatten the list of tuples, i.e. the two views are _interleaved_
            X = list(itertools.chain(*X))

        if len(X) == 0:
            return None

        out = {
            "model_inputs": tb_collate([x["model_inputs"] for x in X]),
            "unit_seqlen": tb_collate([x["unit_seqlen"] for x in X]),
            "unit_ids": np.concatenate([x["unit_ids"] for x in X]).astype(str),
            "probe_ids": np.concatenate([x["probe_ids"] for x in X]).astype(str),
        }
        return out


class FFN(nn.Module):
    def __init__(self, dim: int, mult: int, dropout: float, pre_norm: bool):
        super().__init__()
        self.norm = nn.LayerNorm(dim) if pre_norm else nn.Identity()
        self.in_proj = nn.Linear(dim, 2 * dim * mult)
        self.out_proj = nn.Linear(dim * mult, dim)
        self.dp = nn.Dropout(p=dropout)
        self.act = nn.GELU()

    def forward(self, x):
        y = self.norm(x)
        y, gate = self.in_proj(y).chunk(2, dim=-1)
        y = self.act(gate) * y
        y = self.dp(y)
        return self.out_proj(y)
