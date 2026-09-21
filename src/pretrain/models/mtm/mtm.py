from typing import get_args

import numpy as np
import torch
import torch.nn as nn
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from torch_brain.nn import InfiniteVocabEmbedding
from torch_brain.utils.binning import bin_spikes

from core.dataset import IBLBrainWideBench2026
from core.model import BaseModel
from core.nn import Embedding, tfixup_init_
from core.nn.tokenizer import StringArrayTokenizer
from core.utils.checkpoint import log_incompatible_keys
from core.utils.logger import get_cli_logger
from ibl_bwb_eval.tasks import ReadoutSpec, TargetResolution

from .masker import MtMMaskType


class MtM(BaseModel):
    """Multi-task masked transformer over binned spikes :cite:`mtm`.

    Reference implementation: `IBL_MtM_model <https://github.com/colehurwitz/IBL_MtM_model>`_.

    Transformer encoder-based self-supervised multi-task model for neural population dynamics.

    Extend NDT with multi-task masking objectives:
        - co-smooth: predict a masked neuron from the activity of other neurons.
        - causal: predict future time steps from past context.
        - inter-region: predict one region's activity from other regions.
        - intra-region: predict one neuron from other neurons in the same region.

    Returns predicted firing rates and neural latents.

    **Note:** `MtM` intentionally has a lot of overlap with the `NDTStitch` implementation (i.e., the DRY code principle is not respected).
    This design choice is motivated by the desire to make an individual model fully understandable for a user without having to navigate between models.

    We document/highlight the key differences between `NDTStitch` and `MtM`.

    **Removed:**

    * Nothing. The `NDTStitch` structure is kept whole: per-session in/out stitchers,
      session embedding, encoder and T-Fixup init are unchanged.

    **Added:**

    * **A session-shared projection** after the per-session `in_stitcher`:
      `Linear -> Softsign -> Linear`, taking `shared_proj_input_dim` to `hidden_dim`.
      The per-session layer stays a single linear, so the width added here is paid
      once and not once per session.
    * **Mask token added in the sequence:** one learned embedding per mask mode,
      prepended before the encoder and dropped after it.
    * **Region tokenization:** `input_fn` also emits a `regions` array, which the
      inter-region and intra-region objectives need to pick their target region.

    **Updated:**

    * `in_stitcher`: num_units -> `shared_proj_input_dim`, rather than
      num_units -> `hidden_dim`, since the shared projection now produces `hidden_dim`.
    * `encode()` and `forward()` take a `mask_mode`, and `encode()` drops one extra
      leading token when a mode is active.

    Not tested at any ``context_length`` other than the default (1.0); see
    :ref:`dataset_context_length`.
    """

    def __init__(
        self,
        hidden_dim: int,
        bin_size: float,
        encoder_num_layers: int,
        encoder_dim_feedforward: int,
        encoder_num_heads: int,
        encoder_dropout: float,
        encoder_activation: str,
        pre_encoder_dropout: float,
        post_encoder_dropout: float,
        shared_proj_input_dim: int,
        shared_proj_ffn_mult: int = 2,
        use_custom_init: bool = False,
        tfixup_scale_base: float | None = None,
        tfixup_v_scale_factor: float | None = None,
        finetune_enable: bool = False,
    ):
        super().__init__()

        self.bin_size = bin_size
        self.hidden_dim = hidden_dim
        self.shared_proj_input_dim = shared_proj_input_dim

        # postpone the init of the stitcher to link_datasets
        # as we do not know yet the number of unit per sessions

        self.in_stitcher = nn.ModuleDict()
        self.shared_proj = nn.Sequential(
            nn.Linear(shared_proj_input_dim, shared_proj_input_dim * shared_proj_ffn_mult),
            nn.Softsign(),
            nn.Linear(shared_proj_input_dim * shared_proj_ffn_mult, hidden_dim),
        )

        mask_vocab = list(get_args(MtMMaskType))
        self._mask_token_ids = {m: i for i, m in enumerate(mask_vocab)}
        self.mask_emb = nn.Embedding(len(mask_vocab), hidden_dim)

        self.session_emb = InfiniteVocabEmbedding(hidden_dim)

        encoder_layer = TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=encoder_num_heads,
            dim_feedforward=encoder_dim_feedforward,
            dropout=encoder_dropout,
            activation=encoder_activation,
            norm_first=True,
            batch_first=True,
        )

        # pre-norm layers disable the nested tensor fast path
        self.encoder = TransformerEncoder(
            encoder_layer=encoder_layer,
            num_layers=encoder_num_layers,
            norm=nn.LayerNorm(hidden_dim),
            enable_nested_tensor=False,
        )

        self.pre_encoder_dropout = nn.Dropout(pre_encoder_dropout)
        self.post_encoder_dropout = nn.Dropout(post_encoder_dropout)

        self.out_stitcher = nn.ModuleDict()

        self.tfixup_scale_base = tfixup_scale_base
        self.tfixup_v_scale_factor = tfixup_v_scale_factor

        if use_custom_init:
            self.custom_init()

        self.finetune_enable = finetune_enable

        self.logger = get_cli_logger()

    def link_datasets(
        self,
        train_dataset: IBLBrainWideBench2026,
        val_dataset: IBLBrainWideBench2026,
        test_dataset: IBLBrainWideBench2026 | None = None,
    ):
        brain_regions = train_dataset.get_brain_regions()
        self.region_tokenizer = StringArrayTokenizer(vocab=brain_regions)

        num_unit_per_recording = train_dataset.get_num_unit_per_recording()

        for session_id, num_units in num_unit_per_recording.items():
            self.in_stitcher[session_id] = nn.Linear(num_units, self.shared_proj_input_dim)

            self.out_stitcher[session_id] = nn.Linear(self.hidden_dim, num_units)

        ctx_window = train_dataset.context_window
        self.num_bins = int(ctx_window / self.bin_size)
        self.position_emb = Embedding(self.num_bins, self.hidden_dim)

        self.session_ids = set(train_dataset.get_session_ids()) | set(val_dataset.get_session_ids())
        if test_dataset is not None:
            self.session_ids |= set(test_dataset.get_session_ids())
        self.session_ids = list(self.session_ids)

        if not self.finetune_enable:
            self.session_emb.initialize_vocab(self.session_ids)

    def configure_readout(self, readout_spec: ReadoutSpec):
        self.readout_spec = readout_spec
        self.readout = nn.Linear(self.hidden_dim, readout_spec.dim)

    def input_fn(self, data):
        binned_spikes = bin_spikes(
            spikes=data.spikes,
            num_units=len(data.units),
            bin_size=self.bin_size,
            dtype=np.float32,
        )  # (T, N)

        position = np.arange(self.num_bins, dtype=np.int64)  # (T,)

        session_token = self.session_emb.tokenizer(data.session.id)  # (1,)
        region = self.region_tokenizer.input_fn(data.units.region_cosmos)  # (N,)

        return {
            "model_inputs": {
                "spikes": binned_spikes,
                "positions": position,
                "session_tokens": session_token,
            },
            "regions": region,
        }

    def load_ckpt(self, ckpt: dict):
        state_dict = ckpt["model_state_dict"]
        result = self.load_state_dict(state_dict, strict=False)
        self.logger.info("Loaded pretrained model from checkpoint")
        log_incompatible_keys(result, self.logger)

        self.session_emb.extend_vocab(self.session_ids, exist_ok=True)
        self.session_emb.subset_vocab(self.session_ids)

        self.logger.info("Initialized session vocabularies")
        self.logger.info(f"Number of sessions: {len(self.session_emb.vocab) - 1}")

    def encode(self, spikes, positions, session_tokens, session_id, mask_mode):
        emb = self.in_stitcher[session_id](spikes)
        emb = self.shared_proj(emb)

        emb += self.position_emb(positions)

        # prepend mask token to the sequence
        if mask_mode is not None:
            B = emb.size(0)

            mask_token = torch.tensor([self._mask_token_ids[mask_mode]], device=emb.device)  # (1,)
            mask_emb = self.mask_emb(mask_token)  # (1, D)
            mask_emb = mask_emb.unsqueeze(0).expand(B, -1, -1)  # (B, 1, D)

            emb = torch.cat((mask_emb, emb), dim=1)

        # prepend session token to the sequence
        session_emb = self.session_emb(session_tokens)
        session_emb = session_emb.unsqueeze(1)  # (B, 1, D)
        emb = torch.cat((session_emb, emb), dim=1)

        emb = self.pre_encoder_dropout(emb)
        latent = self.encoder(emb)
        latent = self.post_encoder_dropout(latent)

        latent = latent[:, 1:]  # remove session token
        if mask_mode is not None:
            latent = latent[:, 1:]  # remove prompt/mask token

        return latent

    def forward(self, spikes, positions, session_tokens, mask_mode=None):
        session_token = torch.unique(session_tokens)
        assert len(session_token) == 1, "more than 1 session in the batch"
        session_id = self.session_emb.detokenizer(session_token)

        latent = self.encode(spikes, positions, session_tokens, session_id, mask_mode)

        if hasattr(self, "readout"):
            # predict readout
            if self.readout_spec.target_resolution == TargetResolution.SEQUENCE:
                # (B, T, D) -> (B, 1, D)
                latent = latent.mean(dim=1, keepdim=True)
            pred = self.readout(latent)
        else:
            # predict rates
            pred = self.out_stitcher[session_id](latent)

        return pred

    def custom_init(self):
        """T-Fixup init for the encoder; see :func:`core.nn.tfixup_init_`."""
        tfixup_init_(self.encoder, self.tfixup_scale_base, self.tfixup_v_scale_factor)
