"""Attention dispatch for chained (variable-length) token batches.

``torch_brain``'s rotary attention modules only implement ``forward_varlen`` on top of
xformers' ``BlockDiagonalMask``, and refuse to run when ``use_xformers=False``. These
helpers give the same call signature two interchangeable backends, so a model built on
chained tokens can drop xformers entirely:

``nested`` (default)
    Keep the chained layout, wrap q/k/v as jagged nested tensors and hand them to SDPA's
    varlen kernel. No extra dependency; needs the module's submodules to be applied by hand
    because the rotation cannot run on a jagged tensor.
``xformers``
    Delegate to ``forward_varlen`` (the original kernel). Uses considerably less memory,
    and is somewhat faster, at the cost of the optional ``xformers`` extra and its
    exact-torch-build pin.

Both reach FlashAttention-2 varlen underneath and produce the same values, so the choice is
a runtime one: weights trained under one backend load unchanged into the other.
"""

from importlib.util import find_spec
from typing import Literal, get_args

import torch
import torch.nn.functional as F
from torch_brain.nn import RotaryCrossAttention, RotarySelfAttention, RotaryTimeEmbedding

AttnImpl = Literal["nested", "xformers"]

ATTN_IMPLS: tuple[str, ...] = get_args(AttnImpl)


def validate_attn_impl(impl: str) -> AttnImpl:
    """Narrow a config string to a known backend, raising on anything else."""
    if impl not in ATTN_IMPLS:
        raise ValueError(f"attn_impl={impl!r} is not one of {', '.join(ATTN_IMPLS)}")
    return impl  # type: ignore[return-value]


def uses_xformers(impl: str) -> bool:
    """What to pass as ``use_xformers`` when constructing the attention modules.

    xformers is an optional extra, and torch_brain only notices it is missing on the first
    forward; check here so a mis-provisioned env fails at construction instead of after the
    data pipeline has spun up.
    """
    if validate_attn_impl(impl) != "xformers":
        return False
    if find_spec("xformers") is None:
        raise ImportError(
            "attn_impl='xformers' needs the optional xformers extra: "
            "uv pip install -e \".[train,xformers]\". Use attn_impl='nested' to run on "
            "stock torch instead -- same attention, same checkpoints, somewhat slower."
        )
    return True


def _seqlen_to_offsets(seqlen: torch.Tensor) -> torch.Tensor:
    zero = seqlen.new_zeros(1)
    return torch.cat([zero, seqlen.cumsum(0)]).to(torch.int64)


def _nest(x: torch.Tensor, offsets: torch.Tensor) -> torch.Tensor:
    """``(N_total, H, D) -> (B, H, j, D)`` jagged, ready for SDPA."""
    return torch.nested.nested_tensor_from_jagged(x, offsets=offsets).transpose(1, 2)


def _nested_sdpa(
    *,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    q_pos_emb: torch.Tensor,
    kv_pos_emb: torch.Tensor,
    q_seqlen: torch.Tensor,
    kv_seqlen: torch.Tensor,
    num_heads: int,
    dropout_p: float,
    rotate_value: bool,
) -> torch.Tensor:
    """Rotary attention over chained tokens, jagged all the way through SDPA.

    q/k/v arrive flat as ``(N_total, H*D)``. The rotation runs here, while the tensors are
    still dense: ``RotaryTimeEmbedding.rotate`` unbinds and stacks along the head
    dimension, which a jagged tensor will not take.
    """
    query = query.unflatten(-1, (num_heads, -1))  # (N_q, H, D)
    key = key.unflatten(-1, (num_heads, -1))
    value = value.unflatten(-1, (num_heads, -1))

    query = RotaryTimeEmbedding.rotate(x=query, rotary_emb=q_pos_emb, unsqueeze_dim=1)
    key = RotaryTimeEmbedding.rotate(x=key, rotary_emb=kv_pos_emb, unsqueeze_dim=1)
    if rotate_value:
        value = RotaryTimeEmbedding.rotate(x=value, rotary_emb=kv_pos_emb, unsqueeze_dim=1)

    q_offsets = _seqlen_to_offsets(q_seqlen)
    kv_offsets = _seqlen_to_offsets(kv_seqlen)

    out = F.scaled_dot_product_attention(
        _nest(query, q_offsets),
        _nest(key, kv_offsets),
        _nest(value, kv_offsets),
        dropout_p=dropout_p,
    )
    out = out.transpose(1, 2).values()  # (B, H, j, D) -> (N_q, H, D)

    if rotate_value:
        out = RotaryTimeEmbedding.rotate(
            x=out,
            rotary_emb=RotaryTimeEmbedding.invert(q_pos_emb),
            unsqueeze_dim=1,
        )

    return out.flatten(-2)  # (N_q, H*D)


def cross_attn(
    module: RotaryCrossAttention,
    *,
    x_query: torch.Tensor,
    x_context: torch.Tensor,
    query_pos_emb: torch.Tensor,
    context_pos_emb: torch.Tensor,
    query_seqlen: torch.Tensor,
    context_seqlen: torch.Tensor,
    impl: AttnImpl,
) -> torch.Tensor:
    """``forward_varlen`` for :class:`RotaryCrossAttention`, on either backend."""
    if validate_attn_impl(impl) == "xformers":
        return module.forward_varlen(
            x_query,
            x_context,
            query_pos_emb,
            context_pos_emb,
            query_seqlen,
            context_seqlen,
        )

    x_query = module.norm(x_query)
    x_context = module.norm_context(x_context)
    q = module.to_q(x_query)
    k, v = module.to_kv(x_context).chunk(2, dim=-1)
    out = _nested_sdpa(
        query=q,
        key=k,
        value=v,
        q_pos_emb=query_pos_emb,
        kv_pos_emb=context_pos_emb,
        q_seqlen=query_seqlen,
        kv_seqlen=context_seqlen,
        num_heads=module.heads,
        dropout_p=module.dropout if module.training else 0.0,
        rotate_value=module.rotate_value,
    )
    return module.to_out(out)


def self_attn(
    module: RotarySelfAttention,
    *,
    x: torch.Tensor,
    rotary_time_emb: torch.Tensor,
    x_seqlen: torch.Tensor,
    impl: AttnImpl,
) -> torch.Tensor:
    """``forward_varlen`` for :class:`RotarySelfAttention`, on either backend."""
    if validate_attn_impl(impl) == "xformers":
        return module.forward_varlen(x, rotary_time_emb, x_seqlen)

    x = module.norm(x)
    q, k, v = module.to_qkv(x).chunk(3, dim=-1)
    out = _nested_sdpa(
        query=q,
        key=k,
        value=v,
        q_pos_emb=rotary_time_emb,
        kv_pos_emb=rotary_time_emb,
        q_seqlen=x_seqlen,
        kv_seqlen=x_seqlen,
        num_heads=module.heads,
        dropout_p=module.dropout if module.training else 0.0,
        rotate_value=module.rotate_value,
    )
    return module.to_out(out)
