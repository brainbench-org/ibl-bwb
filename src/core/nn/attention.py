import torch
import torch.nn.functional as F
from torch import nn

try:
    import xformers.ops as xops
except ImportError:
    xops = None


class SelfAttention(nn.Module):
    r"""Multi-Head Self Attention module.

    Args:
        dim: dimension of incoming tokens
        heads: number of heads for multi-head attention
        dim_head: dimension of each head (set to ``heads // dim`` if None)
        dropout: attention dropout probability
    """

    def __init__(
        self,
        dim: int,
        heads: int,
        dim_head: int | None = None,
        dropout: float = 0.0,
    ):
        super().__init__()

        dim_head = dim_head or (dim // heads)
        inner_dim = dim_head * heads
        self.heads = heads
        self.dropout = dropout

        self.norm = nn.LayerNorm(dim)

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)
        self.to_out = nn.Linear(inner_dim, dim)

    def forward(self, x, x_mask=None):
        """Forward pass for fixed-length sequences.

        Shape:
            - x: (B, N, D)
            - x_mask: (B, N, N)
            - Output: (B, N, D)

            where B is batch size, N is sequence length, D is input dimension,
            and D_h is head dimension.
        """
        # normalize
        x = self.norm(x)

        # project to q, k, v
        q, k, v = self.to_qkv(x).chunk(3, dim=-1)

        # select attention kernel
        if xops is not None and x.device.type == "cuda":
            attn_func = attn_xformers_func
        else:
            attn_func = attn_pytorch_func

        # apply attention
        out = attn_func(
            query=q,
            key=k,
            value=v,
            num_heads=self.heads,
            dropout_p=self.dropout if self.training else 0,
            attn_mask=x_mask,
        )

        # project back to dim
        out = self.to_out(out)
        return out

    def forward_varlen(self, x, x_seqlen):
        """Forward pass for variable-length sequences.

        Shape:
            - x: (N_total, D)
            - x_seqlen: (B,)
            - Output: (N_total, D)

            where N_total is the total sequence length across the batch,
            B is batch size, D is input dimension, and D_h is head dimension.
        """
        # normalize
        x = self.norm(x)

        # project to q, k, v
        q, k, v = self.to_qkv(x).chunk(3, dim=-1)

        # select attention kernel
        if xops is None:
            raise RuntimeError("No varlen attention kernel available, please install xformers.")
        if x.device.type != "cuda":
            raise NotImplementedError("No varlen attention kernel available for CPU.")

        # apply attention
        out = attn_xformers_varlen_func(
            query=q,
            key=k,
            value=v,
            num_heads=self.heads,
            dropout_p=self.dropout if self.training else 0,
            q_seqlen=x_seqlen,
            kv_seqlen=None,  # self-attention has the same seqlen for q, k, v
        )

        # project back to dim
        out = self.to_out(out)
        return out


def attn_pytorch_func(
    *,
    query,
    key,
    value,
    attn_mask=None,
    num_heads: int,
    dropout_p: float,
):
    r"""Wraps the default scaled-dot-product attention implementation.

    Args:
        query: The query tensor, with shape (b, n_q, (h d))
        key: The key tensor, with shape (b, n_kv, (h d))
        value: The value tensor, with shape (b, n_kv, (h d))
        num_heads: The number of attention heads
        dropout_p: The dropout probability
        attn_mask: The attention mask, with shape (b, n_kv)

    Returns:
        The output tensor, with shape (b, n_q, (h d))
    """
    # uses the default scaled dot product attention from pytorch
    # https://pytorch.org/docs/stable/generated/torch.nn.functional.scaled_dot_product_attention.html
    # this implements basic versions of memory efficient attention and flash attention
    # but more advanced versions are available in xformers and flash_attn (varlen)
    # which allow us to perform complex masking operations

    # default attention expects shape b h n d
    # (B, N, H*D) -> (B, H, N, D)
    query = query.reshape(query.shape[0], query.shape[1], num_heads, -1).permute(0, 2, 1, 3)
    key = key.reshape(key.shape[0], key.shape[1], num_heads, -1).permute(0, 2, 1, 3)
    value = value.reshape(value.shape[0], value.shape[1], num_heads, -1).permute(0, 2, 1, 3)

    # attention mask
    if attn_mask is not None:
        attn_mask = attn_mask[:, None, None, :]  # (B, 1, 1, N)

    # perform attention, by default will use the optimal attention implementation
    out = F.scaled_dot_product_attention(
        query,
        key,
        value,
        attn_mask=attn_mask,
        dropout_p=dropout_p,
    )

    # (B, H, N, D) -> (B, N, H*D)
    out = out.permute(0, 2, 1, 3).reshape(out.shape[0], out.shape[2], -1)
    return out


def attn_xformers_func(
    *,
    query,
    key,
    value,
    attn_mask=None,
    num_heads: int,
    dropout_p: float,
):
    r"""Wraps the xformers memory efficient attention implementation.

    Args:
        query: The query tensor, with shape (b n (h d))
        key: The key tensor, with shape (b n (h d))
        value: The value tensor, with shape (b n (h d))
        attn_mask: The attention mask, with shape (b, n_kv). A value of True indicates
            that the element should take part in attention.
        num_heads: The number of attention heads
        dropout_p: The dropout probability

    Returns:
        The output tensor, with shape (b n (h d))
    """
    # xformers attention expects shape (1, n, h, d)
    # (B, N, H*D) -> (B, N, H, D)

    query = query.reshape(query.shape[0], query.shape[1], num_heads, -1)
    key = key.reshape(key.shape[0], key.shape[1], num_heads, -1)
    value = value.reshape(value.shape[0], value.shape[1], num_heads, -1)

    # WARNING: this is very slow, avoid using attn_mask if possible, refer to xformers
    # documentation
    if attn_mask is not None:
        attn_mask = attn_mask[:, None, None, :]  # (B, 1, 1, M)
        attn_mask = attn_mask.expand(-1, num_heads, query.size(1), -1)  # (B, H, N, M)
        attn_bias = attn_mask.to(query.dtype).masked_fill(attn_mask.logical_not(), float("-inf"))
    else:
        attn_bias = None

    out = xops.memory_efficient_attention(
        query,
        key,
        value,
        attn_bias=attn_bias,
        p=dropout_p,
    )

    out = out.reshape(out.shape[0], out.shape[1], -1)  # (B, N, H, D) -> (B, N, H*D)
    return out


def attn_xformers_varlen_func(
    *,
    query,
    key,
    value,
    q_seqlen,
    kv_seqlen,
    num_heads: int,
    dropout_p: float,
):
    r"""Wraps the xformers memory efficient attention implementation, for varlen batches.

    Args:
        query: The query tensor, with shape (n, (h d))
        key: The key tensor, with shape (n, (h d))
        value: The value tensor, with shape (n, (h d))
        num_heads: The number of attention heads
        dropout_p: The dropout probability
        q_seqlen: The sequence length of the query tensor
        kv_seqlen: The sequence length of the key and value tensors

    Returns:
        The output tensor, with shape (n, (h d))
    """
    # xformers attention expects shape (1, n, h, d)
    # (N, H*D) -> (1, N, H, D)
    query = query.reshape(1, query.shape[0], num_heads, -1)
    key = key.reshape(1, key.shape[0], num_heads, -1)
    value = value.reshape(1, value.shape[0], num_heads, -1)

    if isinstance(q_seqlen, torch.Tensor):
        q_seqlen = q_seqlen.tolist()
    if isinstance(kv_seqlen, torch.Tensor):
        kv_seqlen = kv_seqlen.tolist()

    # fill attention_bias with BlockDiagonalMask
    with torch.no_grad():
        attn_bias = xops.fmha.BlockDiagonalMask.from_seqlens(
            q_seqlen=q_seqlen,
            kv_seqlen=kv_seqlen,
        )

    out = xops.memory_efficient_attention(
        query,
        key,
        value,
        attn_bias=attn_bias,
        p=dropout_p,
    )

    out = out.reshape(out.shape[1], -1)  # (1, N, H, D) -> (N, H*D)
    return out
