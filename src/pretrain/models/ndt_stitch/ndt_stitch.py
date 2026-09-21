import numpy as np
import torch
import torch.nn as nn
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from torch_brain.nn import InfiniteVocabEmbedding
from torch_brain.utils.binning import bin_spikes

from core.dataset import IBLBrainWideBench2026
from core.model import BaseModel
from core.nn import Embedding, tfixup_init_
from core.utils.checkpoint import log_incompatible_keys
from core.utils.logger import get_cli_logger
from ibl_bwb_eval.tasks import ReadoutSpec, TargetResolution


class NDTStitch(BaseModel):
    """Multi-session NDT with per-session stitchers :cite:`ndt`.

    Reference implementation: `neural-data-transformers
    <https://github.com/snel-repo/neural-data-transformers>`_.

    **Note:** `NDTStitch` intentionally has a lot of overlap with the single session `NDT` implementation (i.e., the DRY code principle is not respected).
    This design choice is motivated by the desire to make an individual model fully understandable for a user without having to navigate between models.

    We document/highlight the key differences between single sesion and it's stitch version.

    **Removed:**
    * The spike embedding strategy.
    * `attn_mask` usage.

    **Added:**
    * **The NDT in/out_stitcher:** For each session of the dataset, an NDT stitcher maps dim_1 to dim_2.

      * `in_stitcher`: num_units -> hidden_dim
      * `out_stitcher`: hidden_dim -> num_units

      Each stitcher is a single linear layer. A two-layer MLP was also available,
      and is not scalable: its hidden width grows with num_units, so the parameter
      count is quadratic in unit count and paid once per session.

    * **Session embeddings** to account for across-session variability.

    **Updated:**
    * The encoder/transformer is initialized in the `__init__()` method and no more in `link_datasets()` as `hidden_dim` is a parameter.

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
        use_custom_init: bool = False,
        tfixup_scale_base: float | None = None,
        tfixup_v_scale_factor: float | None = None,
        finetune_enable: bool = False,
    ):
        super().__init__()

        self.bin_size = bin_size
        self.hidden_dim = hidden_dim

        # postpone the init of the stitcher to link_datasets
        # as we do not know yet the number of unit per sessions
        self.in_stitcher = nn.ModuleDict()

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
        num_unit_per_recording = train_dataset.get_num_unit_per_recording()

        for session_id, num_units in num_unit_per_recording.items():
            self.in_stitcher[session_id] = nn.Linear(num_units, self.hidden_dim)
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

        session_tokens = self.session_emb.tokenizer(data.session.id)  # (1,)

        return {
            "model_inputs": {
                "spikes": binned_spikes,
                "positions": position,
                "session_tokens": session_tokens,
            },
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

    def encode(self, spikes, positions, session_tokens, session_id):
        emb = self.in_stitcher[session_id](spikes)
        emb += self.position_emb(positions)

        # prepend session token to the sequence
        session_emb = self.session_emb(session_tokens)
        session_emb = session_emb.unsqueeze(1)  # (B, 1, D)
        emb = torch.cat((session_emb, emb), dim=1)

        emb = self.pre_encoder_dropout(emb)
        latent = self.encoder(emb)
        latent = self.post_encoder_dropout(latent)

        latent = latent[:, 1:]  # remove session token
        return latent

    def forward(self, spikes, positions, session_tokens):
        session_token = torch.unique(session_tokens)
        assert len(session_token) == 1, "more than 1 session in the batch"
        session_id = self.session_emb.detokenizer(session_token)

        latent = self.encode(spikes, positions, session_tokens, session_id)

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
