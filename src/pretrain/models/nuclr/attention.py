import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.nn.attention.varlen import varlen_attn
from torch_brain.nn.position_embeddings import RotaryTimeEmbedding


def _seqlen_to_cuseq(seqlen: Tensor) -> tuple[Tensor, int]:
    batch_size = len(seqlen)
    cu_seq = torch.zeros(batch_size + 1, device=seqlen.device, dtype=torch.int32)
    cu_seq[1:] = seqlen.cumsum(0)
    max_len: int = seqlen.max().item()  # ty:ignore[invalid-assignment]
    return cu_seq, max_len


class SelfAttention(nn.Module):
    def __init__(
        self,
        dim: int,
        heads: int,
        dim_head: int | None = None,
        to_qkv_bias: bool = True,
        to_out_bias: bool = True,
        pre_norm: bool = True,
        rotate_value: bool = False,
    ):
        super().__init__()
        self.heads = heads
        self.rotate_value = rotate_value
        self.dim_head = dim_head = dim_head or (dim // heads)

        # build networks
        inner_dim = dim_head * heads
        self.norm = nn.LayerNorm(dim) if pre_norm else nn.Identity()
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=to_qkv_bias)
        self.to_out = nn.Linear(inner_dim, dim, bias=to_out_bias)

    def forward(
        self,
        x: Tensor,
        rotary: Tensor | None = None,
        seqlen: Tensor | None = None,
    ):

        q, k, v = self.to_qkv(self.norm(x)).chunk(3, dim=-1)

        if seqlen is not None:
            assert q.ndim == 2 and k.ndim == 2 and v.ndim == 2
            n, h, d = q.size(0), self.heads, self.dim_head
            q = q.view(n, h, d)
            k = k.view(n, h, d)
            v = v.view(n, h, d)
            cuseq, seqmax = _seqlen_to_cuseq(seqlen)

            def attn_func(q, k, v):
                return varlen_attn(q, k, v, cuseq, cuseq, seqmax, seqmax)

        else:
            assert q.ndim == 3 and k.ndim == 3 and v.ndim == 3
            b, n, h, d = q.size(0), q.size(1), self.heads, self.dim_head
            q = q.view(b, n, h, d).permute(0, 2, 1, 3)
            k = k.view(b, n, h, d).permute(0, 2, 1, 3)
            v = v.view(b, n, h, d).permute(0, 2, 1, 3)

            def attn_func(q, k, v):
                return F.scaled_dot_product_attention(q, k, v)

        if rotary is not None:
            q = RotaryTimeEmbedding.rotate(q, rotary, 1)
            k = RotaryTimeEmbedding.rotate(k, rotary, 1)
            if self.rotate_value:
                v = RotaryTimeEmbedding.rotate(v, rotary, 1)

        out = attn_func(q, k, v)
        assert isinstance(out, Tensor)

        if rotary is not None and self.rotate_value:
            rotary_inv = RotaryTimeEmbedding.invert(rotary)
            out = RotaryTimeEmbedding.rotate(out, rotary_inv, 1)

        if seqlen is not None:
            # out is n,h,d
            out = out.view(n, h * d)
        else:
            # out is b,h,n,d
            out = out.permute(0, 2, 1, 3).view(b, n, h * d)

        out = self.to_out(out)
        return out
