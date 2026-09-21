import math
from typing import Any

import numpy as np
import torch
from torch import nn
from torch_brain.batching import pad, track_mask
from torch_brain.data import Data
from torch_brain.nn import InfiniteVocabEmbedding
from torch_brain.utils.binning import bin_spikes

from core.dataset import IBLBrainWideBench2026
from core.model import BaseModel
from core.nn import Embedding
from core.utils.checkpoint import log_incompatible_keys
from core.utils.logger import get_cli_logger
from ibl_bwb_eval.tasks import ReadoutSpec, TargetResolution


def create_attn_mask(
    time_idx: torch.Tensor, num_ctx_tokens: int, n_heads: int, is_causal: bool
) -> torch.Tensor:
    B, L = time_idx.size()
    N_CTX = num_ctx_tokens
    dev = time_idx.device

    if is_causal:
        # Per-sample mask (varies across the batch), which typically disables Flash attention.
        # Non-context tokens use causal masking over non-context tokens.
        attn_mask = torch.zeros((B, N_CTX + L, N_CTX + L), dtype=torch.bool, device=dev)
        attn_mask[:, :N_CTX, N_CTX:] = True
        causal_mask = time_idx[:, :, None] < time_idx[:, None, :]
        attn_mask[:, N_CTX:, N_CTX:] = causal_mask

        # b l l -> (b enc_heads) l l
        attn_mask = attn_mask.unsqueeze(1)
        attn_mask = attn_mask.expand(-1, n_heads, -1, -1)
        attn_mask = attn_mask.reshape(-1, N_CTX + L, N_CTX + L)

    else:
        # Shared mask (same for all samples), which is compatible with Flash attention.
        attn_mask = torch.zeros((N_CTX + L, N_CTX + L), dtype=torch.bool, device=dev)
        attn_mask[:N_CTX, N_CTX:] = True

    return attn_mask


class NDT2(BaseModel):
    """Multi-context masked autoencoder over patched spike tokens :cite:`ndt2`.

    Reference implementation: `context_general_bci
    <https://github.com/joel99/context_general_bci>`_.

    Not tested at any ``context_length`` other than the default (1.0); see
    :ref:`dataset_context_length`.
    """

    SSL_MASK_TOKEN = 0
    BHVR_TOKEN = 1

    def __init__(
        self,
        is_ssl: bool,
        hidden_dim: int,
        units_per_patch: int,
        max_spikes: int,
        max_num_units: int,
        bin_size: float,
        tokenize_session: bool,
        tokenize_subject: bool,
        enc_depth: int,
        enc_heads: int,
        enc_ffn_mult: float,
        dec_depth: int,
        dec_heads: int,
        dec_ffn_mult: float,
        dropout: float,
        activation: str = "gelu",
        pre_norm: bool = True,
        is_causal: bool = False,
        finetune_enable: bool = False,
    ):
        super().__init__()

        self.bin_size = bin_size
        self.hidden_dim = hidden_dim

        if units_per_patch > hidden_dim:
            raise ValueError(
                f"hidden_dim should be greater than units_per_patch (hidden_dim:{hidden_dim}, units_per_patch:{units_per_patch})"
            )
        if hidden_dim % units_per_patch != 0:
            raise ValueError(
                f"hidden_dim should be divisible by units_per_patch (hidden_dim:{hidden_dim}, units_per_patch:{units_per_patch})"
            )

        self.max_num_units = max_num_units

        self.is_ssl = is_ssl
        self.units_per_patch = units_per_patch
        self.max_spikes = max_spikes

        self.num_ctx_tokens = 0

        self.tokenize_session = tokenize_session
        if tokenize_session:
            self.session_emb = InfiniteVocabEmbedding(hidden_dim)
            self.session_bias = nn.Parameter(torch.randn(hidden_dim) / math.sqrt(hidden_dim))
            self.num_ctx_tokens += 1

        self.tokenize_subject = tokenize_subject
        if tokenize_subject:
            self.subject_emb = InfiniteVocabEmbedding(hidden_dim)
            self.subject_bias = nn.Parameter(torch.randn(hidden_dim) / math.sqrt(hidden_dim))
            self.num_ctx_tokens += 1

        assert self.num_ctx_tokens != 0, "At least 1 context token"

        self.max_num_patches = max_num_units // units_per_patch
        self.enc_depth = enc_depth
        self.enc_heads = enc_heads
        self.enc_ffn_mult = enc_ffn_mult
        self.dec_depth = dec_depth
        self.dec_heads = dec_heads
        self.dec_ffn_mult = dec_ffn_mult
        self.dropout = dropout
        self.activation = activation
        self.pre_norm = pre_norm
        self.is_causal = is_causal

        # Learnable token appended for masked-token reconstruction in SSL.
        # Learnable query token used to decode behavior at selected time bins.
        # 0: SSL Mask token, 1: Behavior token
        self.query_emb = Embedding(2, hidden_dim)

        self.finetune_enable = finetune_enable

        self.logger = get_cli_logger()

    def link_datasets(
        self,
        train_dataset: IBLBrainWideBench2026,
        val_dataset: IBLBrainWideBench2026,
        test_dataset: IBLBrainWideBench2026 | None = None,
    ):
        self.session_ids = set(train_dataset.get_session_ids()) | set(val_dataset.get_session_ids())
        if test_dataset is not None:
            self.session_ids |= set(test_dataset.get_session_ids())
        self.session_ids = list(self.session_ids)

        self.subject_ids = set(train_dataset.get_subject_ids()) | set(val_dataset.get_subject_ids())
        if test_dataset is not None:
            self.subject_ids |= set(test_dataset.get_subject_ids())
        self.subject_ids = list(self.subject_ids)

        if self.tokenize_session and not self.finetune_enable:
            self.session_emb.initialize_vocab(train_dataset.get_session_ids())

        if self.tokenize_subject and not self.finetune_enable:
            self.subject_emb.initialize_vocab(train_dataset.get_subject_ids())

        ctx_window = train_dataset.context_window
        self.num_bin = int(ctx_window / self.bin_size)

        self.encoder = NDT2Encoder(
            is_ssl=self.is_ssl,
            num_ctx_tokens=self.num_ctx_tokens,
            max_spikes=self.max_spikes,
            units_per_patch=self.units_per_patch,
            num_bin=self.num_bin,
            max_num_patches=self.max_num_patches,
            hidden_dim=self.hidden_dim,
            depth=self.enc_depth,
            heads=self.enc_heads,
            ffn_mult=self.enc_ffn_mult,
            dropout=self.dropout,
            activation=self.activation,
            pre_norm=self.pre_norm,
            is_causal=self.is_causal,
        )

        self.decoder = NDT2Decoder(
            is_ssl=self.is_ssl,
            num_ctx_tokens=self.num_ctx_tokens,
            num_bin=self.num_bin,
            max_num_patches=self.max_num_patches,
            hidden_dim=self.hidden_dim,
            depth=self.dec_depth,
            heads=self.dec_heads,
            ffn_mult=self.dec_ffn_mult,
            dropout=self.dropout,
            activation=self.activation,
            pre_norm=self.pre_norm,
            is_causal=self.is_causal,
        )

        self.proj = nn.Linear(self.hidden_dim, self.units_per_patch)

    def configure_readout(self, readout_spec: ReadoutSpec):
        self.logger.info(
            f"Configuring readout with {readout_spec.dim} dimensions, overriding final proj"
        )
        self.readout_spec = readout_spec
        self.readout = nn.Linear(self.hidden_dim, readout_spec.dim)

    def load_ckpt(self, ckpt: dict):
        state_dict = ckpt["model_state_dict"]
        if hasattr(self, "readout"):
            state_dict = {k: v for k, v in state_dict.items() if not k.startswith("decoder.")}
        result = self.load_state_dict(state_dict, strict=False)
        self.logger.info("Loaded pretrained model from checkpoint")
        log_incompatible_keys(result, self.logger, ignore=("decoder.",))

        if self.tokenize_session:
            self.session_emb.extend_vocab(self.session_ids, exist_ok=True)
            self.session_emb.subset_vocab(self.session_ids)
            self.logger.info(
                f"Initialized session vocab with {len(self.session_emb.vocab) - 1} sessions"
            )
        if self.tokenize_subject:
            self.subject_emb.extend_vocab(self.subject_ids, exist_ok=True)
            self.subject_emb.subset_vocab(self.subject_ids)
            self.logger.info(
                f"Initialized subject vocab with {len(self.subject_emb.vocab) - 1} subjects"
            )

    def input_fn(self, data: Data) -> dict[str, Any]:
        N = len(data.units.id)
        P = self.units_per_patch

        assert self.max_num_units >= N, f"Increase max_num_units to at least {N}"

        # `self.max_spikes` acts as the padding index.
        spikes_binned = bin_spikes(
            spikes=data.spikes,
            num_units=N,
            bin_size=self.bin_size,
            max_spikes=self.max_spikes - 1,
            dtype=np.int32,
        )  # (T, N)

        N_PATCHES = math.ceil(N / P)

        N_EXTRA = N_PATCHES * P - N

        if N_EXTRA > 0:
            # Pad the final patch when unit count is not divisible by patch size.
            # Example: 5 units with patch size 2 requires 1 padded unit.
            unit_pad = ((0, 0), (0, N_EXTRA))
            spikes_binned = np.pad(
                spikes_binned,
                unit_pad,
                mode="constant",
                constant_values=self.max_spikes,
            )  # (T, N_PATCHES*P), w/ N = N_PATCHES*P - N_EXTRA

        T = spikes_binned.shape[0]

        # Flatten in time-major patch order to match the NDT2 token layout.
        # (T, N_PATCHES*P) -> (T*N_PATCHES, P)
        spikes_patched = spikes_binned.reshape(T, N_PATCHES, P).reshape(-1, P)

        session_idx, subject_idx, task_idx = [], [], []

        if self.tokenize_session:
            session_idx = self.session_emb.tokenizer(data.session.id)  # (1,)
        if self.tokenize_subject:
            subject_idx = self.subject_emb.tokenizer(data.subject.id)  # (1,)

        # Time and space indices for flattened patch tokens.
        time_idx = np.arange(T, dtype=np.int32)
        time_idx = np.repeat(time_idx, N_PATCHES)  # (T,) -> (T*N_PATCHES,)

        space_idx = np.arange(N_PATCHES, dtype=np.int32)
        space_idx = np.tile(space_idx, T)  # (N_PATCHES,) -> (T*N_PATCHES,)

        data_dict = {
            "model_inputs": {
                # Input sequence
                "in_patches": pad(spikes_patched),  # (T*N_PATCHES, P)
                "in_time_idx": pad(time_idx),  # (T*N_PATCHES)
                "in_space_idx": pad(space_idx),  # (T*N_PATCHES)
                "in_not_pad": track_mask(time_idx),  # (T*N_PATCHES)
                # Context tokens
                "session_idx": session_idx,  # (1,)
                "subject_idx": subject_idx,  # (1,)
                "task_idx": task_idx,  # (1,)
            },
        }

        if self.is_ssl:
            # Register validity signal so we knows
            # which binned spikes are real vs extra (spatially padded)
            spikes_binned_is_valid = np.ones((N_PATCHES * P, T), dtype=bool)

            if N_EXTRA > 0:
                # The last patch contains spatially-padded units.
                # Mark it invalid so the loss ignores it.
                spikes_binned_is_valid[-N_EXTRA:] = False

            # (N_PATCHES*P, T) -> (T*N_PATCHES, P)
            spikes_patched_is_valid = (
                spikes_binned_is_valid.reshape(N_PATCHES, P, T).transpose(2, 0, 1).reshape(-1, P)
            )

            data_dict["ssl"] = {
                "target": pad(spikes_patched),  # (T*N_PATCHES, P)
                "is_valid": pad(spikes_patched_is_valid),  # (T*N_PATCHES, P)
                "mask": track_mask(spikes_patched),  # (T*N_PATCHES, P)
            }

            # For SSL queries are not deterministic, query_time_idx, query_idx
            # will be determine at forward() time from masker output

        else:  # For supervised queries are deterministic
            target_resolution = self.readout_spec.target_resolution
            if target_resolution == TargetResolution.SEQUENCE:
                query_time_idx = np.array([T - 1], dtype=np.int32)
            elif target_resolution == TargetResolution.TIMESTEP:
                query_time_idx = np.arange(T, dtype=np.int32)
            else:
                raise ValueError(f"Unrecognized target resolution: {target_resolution}")

            query_idx = np.full_like(query_time_idx, fill_value=NDT2.BHVR_TOKEN)

            data_dict["model_inputs"].update(
                {
                    "query_idx": pad(query_idx),  # (1,) | (T,)
                    "query_time_idx": pad(query_time_idx),  # (1,) | (T,)
                    "query_mask": track_mask(query_time_idx),  # (1,) | (T,)
                }
            )

        return data_dict

    def _pool(
        self,
        latents: torch.Tensor,
        in_not_pad: torch.Tensor,
        in_time_idx: torch.Tensor,
    ):
        """Pool latent patches over space into ``(B, T, D)``."""
        B, _, D = latents.size()
        T = self.num_bin
        dev = latents.device

        # Average-pool latents across spatial patches per time bin.
        # +1 handles padding bucket.
        pooled_latents = torch.zeros((B, T + 1, D), dtype=latents.dtype, device=dev)

        # Route padded entries to the extra bucket.
        index = torch.where(in_not_pad, in_time_idx, T)
        index = index.unsqueeze(-1).expand(-1, -1, D)  # (B, L) -> (B, L, D)
        pooled_latents.scatter_reduce_(
            dim=1, index=index, src=latents, reduce="mean", include_self=False
        )  # (B, L, D) -> (B, T+1, D)
        latents = pooled_latents[:, :-1]  # Drop the padding bucket.

        # Pooling removes the spatial axis, so each time bin is considered present.
        # Padding only occurs in the varying spatial dimension.
        in_not_pad = torch.ones((B, T), dtype=torch.bool, device=dev)

        in_time_idx = torch.arange(self.num_bin, device=dev)
        in_time_idx = in_time_idx.unsqueeze(0).expand(latents.size(0), -1)  # (T,) -> (B, T)

        return latents, in_not_pad, in_time_idx

    def forward(
        self,
        in_patches: torch.Tensor,
        in_time_idx: torch.Tensor,
        in_space_idx: torch.Tensor,
        in_not_pad: torch.Tensor,
        ssl_mask: torch.Tensor | None = None,
        query_idx: torch.Tensor | None = None,
        query_time_idx: torch.Tensor | None = None,
        query_space_idx: torch.Tensor | None = None,
        query_mask: torch.Tensor | None = None,
        session_idx: torch.Tensor | None = None,
        subject_idx: torch.Tensor | None = None,
        task_idx: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:

        if self.is_ssl:
            assert ssl_mask is not None, "Need ssl_mask when doing ssl"
            B, L, P = in_patches.size()
            L_q = int(ssl_mask[0].sum().item())
            L_in = L - L_q

            # Query targets and indices (extracted before in_* are overwritten)
            query_time_idx = in_time_idx[ssl_mask].reshape(B, L_q)
            query_space_idx = in_space_idx[ssl_mask].reshape(B, L_q)
            query_mask = in_not_pad[ssl_mask].reshape(B, L_q)

            # Encoder input, visible tokens only
            in_patches = in_patches[~ssl_mask].reshape(B, L_in, P)
            in_time_idx = in_time_idx[~ssl_mask].reshape(B, L_in)
            in_space_idx = in_space_idx[~ssl_mask].reshape(B, L_in)
            in_not_pad = in_not_pad[~ssl_mask].reshape(B, L_in)

            query_idx = torch.full_like(query_time_idx, fill_value=NDT2.SSL_MASK_TOKEN)

        # Context embeddings
        ctx_tokens = []
        if self.tokenize_session:
            ctx_tokens.append(self.session_emb(session_idx) + self.session_bias)
        if self.tokenize_subject:
            ctx_tokens.append(self.subject_emb(subject_idx) + self.subject_bias)

        ctx_emb = torch.stack(ctx_tokens, dim=1)  # list[(B, H)] -> (B, N_CTX, H)

        # Encode
        latents = self.encoder(
            in_patches, in_time_idx, in_space_idx, in_not_pad, ctx_emb
        )  # (B, L, H)

        # For supervised pool over space
        if not self.is_ssl:
            latents, in_not_pad, in_time_idx = self._pool(latents, in_not_pad, in_time_idx)

        # Append query tokens after encoder latents
        query_tokens = self.query_emb(query_idx)
        latents = torch.cat([latents, query_tokens], dim=1)

        dec_time_idx = torch.cat([in_time_idx, query_time_idx], dim=1)

        dec_space_idx = torch.cat([in_space_idx, query_space_idx], dim=1) if self.is_ssl else None

        dec_padding_mask = torch.cat([~in_not_pad, ~query_mask], dim=1)

        # Decode
        latents = self.decoder(latents, dec_time_idx, dec_space_idx, dec_padding_mask, ctx_emb)

        # Project to Task Output, only query-token outputs in both SSL and supervised modes
        num_query_tokens = query_idx.size(1)
        latents = latents[:, -num_query_tokens:]

        if hasattr(self, "readout"):
            if self.readout_spec.target_resolution == TargetResolution.SEQUENCE:
                # (B, T, D) -> (B, 1, D)
                latents = latents.mean(dim=1, keepdim=True)
            preds = self.readout(latents)
        else:
            preds = self.proj(latents)

        return preds


class NDT2Encoder(nn.Module):
    """Transformer encoder over patched spike tokens with optional context."""

    def __init__(
        self,
        is_ssl: bool,
        num_ctx_tokens: int,
        # Patch and spatiotemporal embedding parameters
        max_spikes: int,
        units_per_patch: int,
        num_bin: int,
        max_num_patches: int,
        # Transformer parameters
        hidden_dim: int,
        depth: int,
        heads: int,
        ffn_mult: float,
        dropout: float,
        activation: str,
        pre_norm: bool,
        is_causal: bool,
    ):
        super().__init__()

        H_DIV_P = hidden_dim // units_per_patch

        self.is_ssl = is_ssl
        self.num_ctx_tokens = num_ctx_tokens

        self.patch_emb = Embedding(max_spikes + 1, H_DIV_P, padding_idx=max_spikes)

        self.time_emb = Embedding(num_bin, hidden_dim)
        self.space_emb = Embedding(max_num_patches, hidden_dim)

        self.is_causal = is_causal
        self.heads = heads

        self.enc_dropout_in = nn.Dropout(dropout)
        self.enc_dropout_out = nn.Dropout(dropout)

        layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=heads,
            dim_feedforward=int(hidden_dim * ffn_mult),
            dropout=dropout,
            batch_first=True,
            activation=activation,
            norm_first=pre_norm,
        )
        # supplying both masks disables the nested tensor fast path
        self.encoder = nn.TransformerEncoder(layer, depth, enable_nested_tensor=False)

    def forward(
        self,
        spikes_patched: torch.Tensor,
        time_idx: torch.Tensor,
        space_idx: torch.Tensor,
        mask: torch.Tensor,
        ctx_emb: torch.Tensor | None,
    ) -> torch.Tensor:
        """Encode input sequence to latents of shape ``(B, L_in, H)``."""
        B = spikes_patched.size(0)
        N_CTX = self.num_ctx_tokens
        dev = spikes_patched.device

        inputs = self.patch_emb(spikes_patched)
        inputs = inputs.flatten(-2)  # (B, L, P, H/P) -> (B, L, H)
        inputs = self.enc_dropout_in(inputs)

        inputs = inputs + self.time_emb(time_idx) + self.space_emb(space_idx)

        if ctx_emb is not None:
            inputs = torch.cat([ctx_emb, inputs], dim=1)

        enc_padding_mask = ~mask
        if N_CTX > 0:
            ctx_padding = torch.zeros((B, N_CTX), dtype=torch.bool, device=dev)
            enc_padding_mask = torch.cat([ctx_padding, enc_padding_mask], dim=1)

        attn_mask = create_attn_mask(time_idx, N_CTX, self.heads, self.is_causal)

        latents = self.encoder(inputs, src_key_padding_mask=enc_padding_mask, mask=attn_mask)
        latents = latents[:, N_CTX:]  # Drop prepended context
        latents = self.enc_dropout_out(latents)

        return latents


class NDT2Decoder(nn.Module):
    """MAE-style decoder for masked reconstruction or supervised readout."""

    def __init__(
        self,
        is_ssl: bool,
        num_ctx_tokens: int,
        # Spatiotemporal embedding parameters
        num_bin: int,
        max_num_patches: int,
        # Transformer parameters
        hidden_dim: int,
        depth: int,
        heads: int,
        ffn_mult: float,
        dropout: float,
        activation: str,
        pre_norm: bool,
        is_causal: bool,
    ):
        super().__init__()

        self.is_ssl = is_ssl
        self.num_ctx_tokens = num_ctx_tokens

        self.dec_time_emb = Embedding(num_bin, hidden_dim)
        if is_ssl:
            # SSL decoder keeps spatial tokens; supervised path spatially pools latents first.
            self.dec_space_emb = Embedding(max_num_patches, hidden_dim)

        # Decoder
        self.is_causal = is_causal
        self.heads = heads

        self.dec_dropout_in = nn.Dropout(dropout)
        self.dec_dropout_out = nn.Dropout(dropout)

        layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=heads,
            dim_feedforward=int(hidden_dim * ffn_mult),
            dropout=dropout,
            activation=activation,
            batch_first=True,
            norm_first=pre_norm,
        )
        # supplying both masks disables the nested tensor fast path
        self.decoder = nn.TransformerEncoder(layer, depth, enable_nested_tensor=False)

    def forward(
        self,
        latents: torch.Tensor,
        time_idx: torch.Tensor,
        space_idx: torch.Tensor | None,
        padding_mask: torch.Tensor,
        ctx_emb: torch.Tensor,
    ) -> torch.Tensor:
        """Decode latents into output features with shape ``(B, L_dec, H)``."""
        B = latents.size(0)
        N_CTX = self.num_ctx_tokens
        dev = latents.device

        latents = self.dec_dropout_in(latents)

        latents = latents + self.dec_time_emb(time_idx)

        if self.is_ssl:
            latents = latents + self.dec_space_emb(space_idx)

        # Prepend context tokens to decoder inputs
        if N_CTX > 0:
            if not self.is_ssl:
                # Keep context embeddings fixed during supervised finetuning
                ctx_emb = ctx_emb.detach()

            latents = torch.cat([ctx_emb, latents], dim=1)

            ctx_padding = torch.zeros((B, N_CTX), dtype=torch.bool, device=dev)
            padding_mask = torch.cat([ctx_padding, padding_mask], dim=1)

        dec_attn_mask = create_attn_mask(time_idx, N_CTX, self.heads, self.is_causal)

        dec_output = self.decoder(latents, src_key_padding_mask=padding_mask, mask=dec_attn_mask)

        output = dec_output[:, N_CTX:]  # Remove prepended context tokens
        output = self.dec_dropout_out(output)
        return output
