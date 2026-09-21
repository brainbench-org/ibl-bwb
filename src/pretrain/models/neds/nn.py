import torch
import torch.nn as nn
import torch.nn.functional as F


# Copied from hf Llama
# Precompute cos and sin for RoPE
def get_cos_sin(dim, max_F, base=10_000, dtype=None, device=None):
    if dtype is None:
        dtype = torch.get_default_dtype()

    inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float().to(device) / dim))
    t = torch.arange(max_F, device=device, dtype=inv_freq.dtype)
    freqs = torch.einsum("i,j->ij", t, inv_freq)

    emb = torch.cat((freqs, freqs), dim=-1)
    return emb.cos().to(dtype), emb.sin().to(dtype)


# Rotates half the hidden dims of the input.
def rotate_half(x):
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), -1)


# Applies RoPE to the query and key tensors.
def apply_rotary_pos_emb(q, k, pos_ids, cos, sin, unsqueeze_dim=1):

    cos = cos[pos_ids].unsqueeze(unsqueeze_dim)
    sin = sin[pos_ids].unsqueeze(unsqueeze_dim)

    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)

    return q_embed, k_embed


class Attention(nn.Module):
    def __init__(
        self,
        hidden_dim,
        num_heads,
        dropout,
        use_rope=False,
        base=10_000.0,
        max_F=100.0,
        n_mod=2,
    ):
        super().__init__()

        self.hidden_dim = hidden_dim
        self.n_heads = num_heads
        assert self.hidden_dim % self.n_heads == 0, "Hidden dim is not multiple of head size"
        self.head_size = self.hidden_dim // self.n_heads

        # Attention parameters
        self.query = nn.Linear(hidden_dim, hidden_dim)
        self.key = nn.Linear(hidden_dim, hidden_dim)
        self.value = nn.Linear(hidden_dim, hidden_dim)

        torch.backends.cuda.enable_flash_sdp(True)
        self.attn_dropout = dropout

        self.dropout = nn.Dropout(dropout)
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)

        # RoPE parameters
        self.use_rope = use_rope
        if self.use_rope:
            cos, sin = get_cos_sin(
                self.head_size,
                max_F * n_mod,
                base=base,
                dtype=self.query.weight.dtype,
                device=self.query.weight.device,
            )
            self.register_buffer("cos", cos, persistent=False)
            self.register_buffer("sin", sin, persistent=False)

    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor | None = None,
        timestamp: torch.Tensor | None = None,  # (B, T)
    ) -> torch.Tensor:
        """
        mask: (B, T) bool key padding mask, True = attend, False = ignore.
              Prevents all queries from attending to masked key positions.
        """

        B, T, _ = x.size()

        if mask is not None:
            # (B, L) -> (B, 1, 1, L) to broadcast over heads and query positions
            mask = mask[:, None, None, :]

        # Compute query, key, value for attention
        q = self.query(x).view(B, T, self.n_heads, self.head_size).transpose(1, 2)
        k = self.key(x).view(B, T, self.n_heads, self.head_size).transpose(1, 2)
        v = self.value(x).view(B, T, self.n_heads, self.head_size).transpose(1, 2)

        # Apply rotations to encode relative positions
        if self.use_rope:
            q, k = apply_rotary_pos_emb(
                q, k, timestamp, self.cos, self.sin, 1
            )  # (B,n_heads,T,head_size)

        out = F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=mask,
            dropout_p=(self.attn_dropout if self.training else 0.0),
            is_causal=False,
        )
        out = out.transpose(1, 2).contiguous().view(B, T, self.hidden_dim)

        return self.out_proj(self.dropout(out))


class TransformerEncoderLayer(nn.Module):
    def __init__(
        self,
        hidden_dim,
        num_heads,
        dropout,
        ffn_factor,
        num_layers,
        use_rope=True,
    ):
        super().__init__()

        self.ln1 = nn.LayerNorm(hidden_dim)

        self.attn = Attention(
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            use_rope=use_rope,
        )

        self.ln2 = nn.LayerNorm(hidden_dim)

        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * ffn_factor),
            nn.Softsign(),
            nn.Linear(hidden_dim * ffn_factor, hidden_dim),
            nn.Dropout(dropout),
        )

        self._fixup_initialization(num_layers)

    def forward(
        self,
        x: torch.Tensor,
        timestamp: torch.Tensor | None = None,
        key_padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:

        x = x + self.attn(self.ln1(x), mask=key_padding_mask, timestamp=timestamp)

        x = x + self.mlp(self.ln2(x))

        return x

    def _fixup_initialization(self, n_layers):
        scale = 0.67 * (n_layers ** (-1.0 / 4.0))
        with torch.no_grad():
            for name, param in self.named_parameters():
                if name.endswith("_proj.weight"):
                    param.mul_(scale)
                elif name.endswith("value.weight"):
                    param.mul_(scale * (2**0.5))
