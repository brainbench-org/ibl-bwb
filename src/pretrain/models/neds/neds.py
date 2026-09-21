import math

import numpy as np
import torch
import torch.nn as nn
from torch_brain.batching import pad
from torch_brain.nn import InfiniteVocabEmbedding
from torch_brain.utils.binning import bin_spikes

from core.dataset import IBLBrainWideBench2026
from core.model import BaseModel
from core.nn import Embedding
from core.utils.checkpoint import log_incompatible_keys
from core.utils.logger import get_cli_logger
from ibl_bwb_eval.tasks import TargetResolution, get_ts1_readout_spec, get_ts1_supported_tasks

from .nn import TransformerEncoderLayer

BEHAVIOR_MODALITIES = get_ts1_supported_tasks()
SEQUENCE_MODALITIES = get_ts1_supported_tasks(target_resolution=TargetResolution.SEQUENCE)
TIMESTEP_MODALITIES = get_ts1_supported_tasks(target_resolution=TargetResolution.TIMESTEP)
ALL_MODALITIES = ["spikes", *BEHAVIOR_MODALITIES]


class SessionLinear(nn.Module):
    """Per-session linear projection using stacked weights applied via bmm.

    Equivalent to one nn.Linear per session, but fully vectorized:
    all sessions in the batch are projected in a single bmm call.
    """

    def __init__(self, n_sessions: int, in_dim: int, out_dim: int):
        super().__init__()
        w = torch.empty(n_sessions * out_dim, in_dim)
        nn.init.kaiming_uniform_(w, a=math.sqrt(5))
        self.weight = nn.Parameter(w.reshape(n_sessions, out_dim, in_dim))

        bound = 1 / math.sqrt(in_dim) if in_dim > 0 else 0
        # Named `bias` on purpose: stacking the sessions makes it 2-D, so `no_weight_decay`
        # spares it by name and not by the 1-D shape rule. Rename it and decay comes back.
        self.bias = nn.Parameter(torch.empty(n_sessions, out_dim).uniform_(-bound, bound))

    def forward(self, x: torch.Tensor, sess_idx: torch.Tensor) -> torch.Tensor:
        """
        x:        (B, T, in_dim)
        sess_idx: (B,)
        returns:  (B, T, out_dim)
        """
        w = self.weight[sess_idx]  # (B, out_dim, in_dim)
        b = self.bias[sess_idx]  # (B, out_dim)
        return torch.bmm(x, w.transpose(-1, -2)) + b.unsqueeze(1)


class NEDSTokenizer(nn.Module):
    def __init__(
        self,
        modality,
        session_ids,
        seq_len,
        hidden_dim,
        dropout,
        max_num_units,
        init_session_vocab=True,
    ):
        super().__init__()

        self.hidden_dim = hidden_dim
        self.modality = modality
        self.seq_len = seq_len

        num_sessions = len(session_ids)
        self._session_to_idx = {str(sid): i for i, sid in enumerate(session_ids)}

        # a behavior modality arrives as one scalar per timestep, spikes as one per unit
        if modality in TIMESTEP_MODALITIES or modality in SEQUENCE_MODALITIES:
            in_dim = 1
        else:  # spikes
            in_dim = max_num_units

        self.in_stitcher = SessionLinear(num_sessions, in_dim, hidden_dim)
        if modality in SEQUENCE_MODALITIES:
            self.in_stitcher = SessionLinear(num_sessions, in_dim, seq_len * hidden_dim)

        # token dropout
        self.pre_enc_dropout = nn.Dropout(dropout)

        self.modality_emb = nn.Parameter(torch.randn(hidden_dim))
        self.session_emb = InfiniteVocabEmbedding(hidden_dim)
        if init_session_vocab:
            self.session_emb.initialize_vocab(session_ids)
        self.position_emb = Embedding(seq_len, hidden_dim)

    def forward(self, x, positions, session_ids):
        """
        inputs: (B, T, D_in) / (B, T, N) for spikes / (B, T) for a scalar modality,
        positions: (B, T)
        session_ids: (B)
        """
        B, T = positions.size()
        dev = x.device

        # bmm requires 3D; a scalar modality arrives 2D after collation, either layout
        if x.dim() == 2:
            x = x.unsqueeze(-1)  # (B, T) -> (B, T, 1)

        ### STITCHING: (B, T, D_in) -> (B, T, D) ###
        sess_idx = [self._session_to_idx[str(sid)] for sid in session_ids]
        sess_idx = torch.tensor(sess_idx, device=dev)  # (B,)

        x = self.in_stitcher(x, sess_idx)  # (B, T, H)

        if self.modality in SEQUENCE_MODALITIES:
            # one target per window, so the stitcher emits the whole sequence at once
            x = x.squeeze(1)  # (B, 1, T*H) -> (B, T*H)
            x = x.reshape(x.shape[0], self.seq_len, -1)  # (B, T*H) -> (B, T, H)

        x = self.pre_enc_dropout(x)  # (B, T, D)

        ### EMBEDDING: (B, T, D) ###
        # (D) -> (B, T, D)
        modality_emb = self.modality_emb[None, None, :].expand(B, T, -1)

        # (B, T, D)
        position_emb = self.position_emb(positions)

        # (B) -> (B, D) -> (B, T, D)
        session_token = self.session_emb.tokenizer(session_ids)
        session_token = torch.tensor(session_token, device=dev)
        session_emb = self.session_emb(session_token)
        session_emb = session_emb[:, None, :].expand(-1, T, -1)

        x_embed = modality_emb + position_emb + session_emb  # (B, T, D)

        return x, x_embed


class NEDSDecoder(nn.Module):
    def __init__(
        self,
        modality,
        session_ids,
        seq_len,
        hidden_dim,
        out_dim,
    ):
        super().__init__()
        self.modality = modality
        self.hidden_dim = hidden_dim
        self.out_dim = out_dim

        num_sessions = len(session_ids)
        self._session_to_idx = {str(sid): i for i, sid in enumerate(session_ids)}

        if self.modality in SEQUENCE_MODALITIES:
            self.seq_pooling_weights = nn.Parameter(torch.rand(num_sessions, seq_len))

        self.out_stitcher = SessionLinear(num_sessions, hidden_dim, out_dim)

    def forward(self, x, session_ids):
        """
        x: (B, T, D)
        session_ids: (B)
        """
        dev = x.device

        sess_idx = [self._session_to_idx[str(sid)] for sid in session_ids]
        sess_idx = torch.tensor(sess_idx, device=dev)  # (B,)

        if self.modality in SEQUENCE_MODALITIES:
            D = x.size(-1)

            pooling_w = self.seq_pooling_weights[sess_idx]  # (B, T)
            pooling_w = pooling_w[:, :, None].expand(-1, -1, D)  # (B, T, D)

            x = torch.sum(x * pooling_w, dim=1, keepdim=True)  # (B, 1, D)

        x = self.out_stitcher(x, sess_idx)

        return x  # (B, 1, D_out) | (B, T, D_out)


class NEDS(BaseModel):
    """Multimodal masked transformer over spikes and behavior :cite:`neds`.

    Reference implementation: `NEDS <https://github.com/yzhang511/NEDS>`_.

    This differs from the reference in one respect: the per-session stitchers, in and
    out, are single linear layers. The two-layer MLP the reference uses was dropped as
    not scalable: its hidden layer is paid once per session and per modality, which on
    the output side alone comes to 250M parameters over the 423 pretraining sessions,
    next to a 9.5M encoder.

    A run is made unimodal by its modalities and the masker's ``mask_types``, not by a
    mode flag: that is how :class:`ts1.models.pretrained.NEDSEvalTrainer` decodes.

    Not tested at any ``context_length`` other than the default (1.0); see
    :ref:`dataset_context_length`.
    """

    def __init__(
        self,
        bin_size,
        hidden_dim,
        tokenizer_dropout,
        encoder_num_layers,
        encoder_num_heads,
        encoder_dropout,
        encoder_ffn_factor,
        modalities=ALL_MODALITIES,
        finetune_enable: bool = False,
    ):
        super().__init__()
        self.logger = get_cli_logger()
        self.bin_size = bin_size

        self.modalities = modalities
        self.hidden_dim = hidden_dim

        self.tokenizers = nn.ModuleDict()
        self.tokenizer_dropout = tokenizer_dropout

        self.mask_emb = nn.Parameter(torch.rand(hidden_dim))

        self.encoder = nn.ModuleList(
            [
                TransformerEncoderLayer(
                    hidden_dim=hidden_dim,
                    num_heads=encoder_num_heads,
                    dropout=encoder_dropout,
                    ffn_factor=encoder_ffn_factor,
                    num_layers=encoder_num_layers,
                    use_rope=True,
                )
                for _ in range(encoder_num_layers)
            ]
        )
        self.encoder_norm = nn.LayerNorm(hidden_dim)

        self.decoders = nn.ModuleDict()

        self.finetune_enable = finetune_enable

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

        num_unit_per_recording = train_dataset.get_num_unit_per_recording()
        max_num_units = max(num_unit_per_recording.values())
        self.max_num_units = max_num_units

        ctx_window = train_dataset.context_window
        self.seq_len = int(ctx_window / self.bin_size)

        for modality in self.modalities:
            self.tokenizers[modality] = NEDSTokenizer(
                modality=modality,
                session_ids=self.session_ids,
                seq_len=self.seq_len,
                hidden_dim=self.hidden_dim,
                max_num_units=max_num_units,
                dropout=self.tokenizer_dropout,
                init_session_vocab=not self.finetune_enable,
            )

            if modality == "spikes":
                self.out_dim = max_num_units
            else:
                self.out_dim = get_ts1_readout_spec(modality).dim

            self.decoders[modality] = NEDSDecoder(
                modality=modality,
                session_ids=self.session_ids,
                seq_len=self.seq_len,
                hidden_dim=self.hidden_dim,
                out_dim=self.out_dim,
            )

    def load_ckpt(self, ckpt: dict):
        state_dict = ckpt["model_state_dict"]

        # filter the session-specific params
        _session_keys = (
            "in_stitcher",
            "out_stitcher",
            "seq_pooling_weights",
        )
        filtered = {k: v for k, v in state_dict.items() if not any(s in k for s in _session_keys)}
        result = self.load_state_dict(filtered, strict=False)
        self.logger.info("Loaded pretrained model from checkpoint")
        log_incompatible_keys(result, self.logger, ignore=_session_keys)

        num_sessions = len(self.session_ids)

        # reinitializing tokenizers
        for tokenizer in self.tokenizers.values():
            in_dim = 1 if tokenizer.modality != "spikes" else self.max_num_units

            tokenizer.in_stitcher = SessionLinear(num_sessions, in_dim, self.hidden_dim)
            if tokenizer.modality in SEQUENCE_MODALITIES:
                tokenizer.in_stitcher = SessionLinear(
                    num_sessions, in_dim, self.seq_len * self.hidden_dim
                )

            tokenizer.session_emb.extend_vocab(self.session_ids, exist_ok=True)
            tokenizer.session_emb.subset_vocab(self.session_ids)
            tokenizer._session_to_idx = {str(sid): i for i, sid in enumerate(self.session_ids)}

        self.logger.info(f"Reinitialized tokenizers for {num_sessions} sessions.")

        # reinitializing decoders
        for decoder in self.decoders.values():
            if decoder.modality == "spikes":
                out_dim = self.max_num_units
            else:
                out_dim = get_ts1_readout_spec(decoder.modality).dim

            decoder.out_stitcher = SessionLinear(num_sessions, self.hidden_dim, out_dim)

            if decoder.modality in SEQUENCE_MODALITIES:
                decoder.seq_pooling_weights = nn.Parameter(torch.rand(num_sessions, self.seq_len))

            self.logger.info(
                f"Reinitialized decoder '{decoder.modality}' for {num_sessions} sessions."
            )

    def input_fn(self, data):
        inputs = {}
        keep_masks = {}
        for modality in self.modalities:
            if modality == "spikes":
                binned_spikes = bin_spikes(
                    spikes=data.spikes,
                    num_units=len(data.units),
                    bin_size=self.bin_size,
                    dtype=np.float32,
                )  # (T, N)

                T, N = binned_spikes.shape
                pad_size = self.max_num_units - N

                # Pad the units dimension (N) to max_num_units.
                binned_spikes = np.pad(
                    binned_spikes,
                    pad_width=((0, 0), (0, pad_size)),
                    mode="constant",
                    constant_values=0.0,
                )

                keep_mask = np.ones((T, self.max_num_units), dtype=bool)
                if pad_size > 0:
                    keep_mask[:, N:] = False

                inputs["spikes"] = binned_spikes
                keep_masks["spikes"] = keep_mask

            else:
                modality_key = get_ts1_readout_spec(modality).value_key

                _absent_shape = (T, 1) if modality in TIMESTEP_MODALITIES else (1, 1)

                if not data.has_nested_attribute(modality_key):
                    input = np.full(_absent_shape, fill_value=-1.0, dtype=np.float32)
                    keep_mask = np.zeros((T, 1), dtype=bool)

                else:
                    input = data.get_nested_attribute(modality_key)

                    if input.shape[0] == 0:
                        input = np.full(_absent_shape, fill_value=-1.0, dtype=np.float32)
                        keep_mask = np.zeros((T, 1), dtype=bool)

                    else:
                        keep_mask = np.ones((T, 1), dtype=bool)

                        if modality in TIMESTEP_MODALITIES:
                            input = input.reshape(-1, 1)  # (T,) -> (T, 1)
                            nan_mask = np.isnan(input).any(axis=-1)  # (T,)
                            if nan_mask.any():
                                keep_mask[nan_mask] = False
                                input = np.where(nan_mask[:, None], 0.0, input)
                        else:  # sequence-level: ensure 2D (1, D)
                            input = input.reshape(1, -1)

                        input = input.astype(np.float32)

                inputs[modality] = pad(input)
                keep_masks[modality] = pad(keep_mask)

        return {
            "model_inputs": {
                **inputs,  # (T, N) | (T, D_out) | (1, D_out)
                "positions": np.arange(self.seq_len, dtype=np.int64),
                "session_ids": data.session.id,
            },
            "keep_masks": {
                **keep_masks,  # (T, 1) | # (T, N)
            },
        }

    def forward(self, model_inputs, keep_masks, modality_masks):
        K = len(self.modalities)

        session_ids = np.array(model_inputs["session_ids"])  # (B)
        positions = model_inputs["positions"]  # (B, T)

        ### MODALITY TOKENIZERS ###
        # create 1 seq from all modalities
        latents, timestamps, key_padding_mask = [], [], []
        for modality in self.modalities:
            _inputs = model_inputs[modality]  # (B, 1, 1) | (B, T, D_in) | (B, T, N)
            _positions = positions.clone()  # (B, T)

            tokens, embeddings = self.tokenizers[modality](
                _inputs, _positions, session_ids
            )  # (B, T, D)

            _keep_mask = keep_masks[modality].any(-1, keepdim=True)
            should_mask = modality_masks[modality] | ~_keep_mask  # (B, T, 1)
            mask_emb = self.mask_emb.to(tokens.dtype)  # (D)
            tokens = torch.where(should_mask, mask_emb, tokens)  # (B, T, D)

            _latents = tokens + embeddings  # (B, T, D)

            latents.append(_latents)
            timestamps.append(_positions)
            key_padding_mask.append(_keep_mask[:, :, 0])  # (B, T)

        latents = torch.cat(latents, dim=1)  # (B, L, D) with L = K*T
        timestamps = torch.cat(timestamps, dim=1)  # (B, L)
        key_padding_mask = torch.cat(key_padding_mask, dim=1)  # (B, K*T)

        ### ENCODER ###
        for layer in self.encoder:
            latents = layer(
                latents, timestamp=timestamps, key_padding_mask=key_padding_mask
            )  # (B, L, D)
        latents = self.encoder_norm(latents)

        chunked_latents = torch.chunk(latents, K, dim=1)  # tuple[(B, T, D)]

        ### DECODER ###
        preds = {}
        for i, modality in enumerate(self.modalities):
            _latents = chunked_latents[i]  # (B, T, D)
            preds[modality] = self.decoders[modality](
                _latents, session_ids
            )  # (B, 1, 1) / (B, T, 1) / (B, T, N)

        return preds
