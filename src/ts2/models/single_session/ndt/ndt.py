import math

import numpy as np
import optuna
import torch
import torch.nn as nn
from omegaconf import DictConfig
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from torch_brain.data import Data
from torch_brain.utils.binning import bin_spikes

from core.model import BaseModel
from core.nn import Embedding, tfixup_init_
from ts2.ts2_dataset import IBLBrainWideBenchTS2


class NDT(BaseModel):
    """Single-session Transformer encoder over binned spikes :cite:`ndt`.

    Reference implementation: `neural-data-transformers
    <https://github.com/snel-repo/neural-data-transformers>`_.

    :meth:`link_datasets` must be called first: the hidden dimension is
    ``num_units * max(unit_emb_dim, 1)``, so the encoder is built there.

    Not tested at any ``context_length`` other than the default (1.0); see
    :ref:`dataset_context_length`.

    Args:
        max_spikes: Size of the count embedding table; counts are clipped to one below it.
        unit_emb_dim: Per-unit count embedding width. ``0`` feeds raw counts, the NDT default.
        encoder_num_heads: Number of attention heads.
        encoder_ffn_factor: Feed-forward sublayer width multiplier relative to the hidden dim.
        encoder_num_layers: Number of Transformer encoder layers.
        encoder_dropout: Dropout inside each encoder layer.
        pre_encoder_dropout: Dropout applied to embeddings before the encoder.
        post_encoder_dropout: Dropout applied to encoder output.
        encoder_activation: Activation of the feed-forward sublayer.
        use_custom_init: If ``True``, apply T-Fixup weight initialisation (see :meth:`custom_init`).
        initrange: Uniform init range for non-Transformer weights.
        tfixup_scale_base: T-Fixup base scale factor (required when ``use_custom_init=True``).
        tfixup_v_scale_factor: Additional scale applied to value weights
            (required when ``use_custom_init=True``).
    """

    def __init__(
        self,
        max_spikes: int,
        unit_emb_dim: int,
        encoder_num_heads: int,
        encoder_ffn_factor: int,
        encoder_num_layers: int,
        encoder_dropout: float,
        pre_encoder_dropout: float,
        post_encoder_dropout: float,
        encoder_activation: str,
        use_custom_init: bool = False,
        initrange: float = 0.1,
        tfixup_scale_base: float | None = None,
        tfixup_v_scale_factor: float | None = None,
    ):
        super().__init__()
        self.max_spikes = max_spikes
        self.unit_emb_dim = unit_emb_dim

        # postpone the init of the encoder to link_datasets
        # as we do not know yet the number of unit
        self.encoder_num_heads = encoder_num_heads
        self.encoder_ffn_factor = encoder_ffn_factor
        self.encoder_dropout = encoder_dropout
        self.encoder_activation = encoder_activation
        self.encoder_num_layers = encoder_num_layers

        self.pre_encoder_dropout = nn.Dropout(pre_encoder_dropout)
        self.post_encoder_dropout = nn.Dropout(post_encoder_dropout)

        self.use_custom_init = use_custom_init
        self.initrange = initrange
        self.tfixup_scale_base = tfixup_scale_base
        self.tfixup_v_scale_factor = tfixup_v_scale_factor

    def link_datasets(
        self,
        train_dataset: IBLBrainWideBenchTS2,
        val_dataset: IBLBrainWideBenchTS2,
        test_dataset: IBLBrainWideBenchTS2 | None = None,
    ):
        num_unit_per_recording = train_dataset.get_num_unit_per_recording()
        assert len(num_unit_per_recording) == 1, (
            "NDT is single session model, use NDT-stitch for multi sessions"
        )

        self.num_units = next(iter(num_unit_per_recording.values()))

        hidden_dim = self.num_units * max(self.unit_emb_dim, 1)

        # unit_emb_dim == 0 is the NDT default: raw counts, no embedding
        if self.unit_emb_dim != 0:
            self.spike_emb = Embedding(self.max_spikes, self.unit_emb_dim)
        self.emb_scale = math.sqrt(hidden_dim)

        ctx_window = train_dataset.context_window
        self.bin_size = train_dataset.BIN_SIZE
        self.num_bins = int(ctx_window / self.bin_size)
        self.position_emb = Embedding(self.num_bins, hidden_dim)

        encoder_layer = TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=self.encoder_num_heads,
            dim_feedforward=hidden_dim * self.encoder_ffn_factor,
            dropout=self.encoder_dropout,
            activation=self.encoder_activation,
            norm_first=True,
            batch_first=True,
        )

        # pre-norm layers disable the nested tensor fast path
        self.encoder = TransformerEncoder(
            encoder_layer=encoder_layer,
            num_layers=self.encoder_num_layers,
            norm=nn.LayerNorm(hidden_dim),
            enable_nested_tensor=False,
        )

        self.readout_layer = nn.Linear(hidden_dim, self.num_units)

        if self.use_custom_init:
            self.custom_init()

    def input_fn(self, data: Data) -> dict:
        bin_counts = bin_spikes(
            data.spikes,
            num_units=len(data.units),
            bin_size=self.bin_size,
            max_spikes=self.max_spikes - 1,
            dtype=np.float32,
        )

        positions = np.arange(self.num_bins, dtype=np.int64)

        return {
            "model_inputs": {
                "spikes": bin_counts,  # (T, N)
                "positions": positions,  # (T)
            },
        }

    def forward(self, spikes, positions):
        spikes = spikes.nan_to_num(0.0)

        if self.unit_emb_dim == 0:
            spike_emb = spikes
        else:
            spike_emb = self.spike_emb(spikes.long())
            spike_emb = spike_emb.flatten(-2)  # (B, T, N, D) -> (B, T, N*D)
        spike_emb = spike_emb * self.emb_scale

        emb = spike_emb + self.position_emb(positions)
        emb = self.pre_encoder_dropout(emb)

        latent = self.encoder(emb)
        latent = self.post_encoder_dropout(latent)

        pred_rates = self.readout_layer(latent)

        return pred_rates

    def custom_init(self):
        """T-Fixup init for the encoder; see :func:`core.nn.tfixup_init_`."""
        tfixup_init_(self.encoder, self.tfixup_scale_base, self.tfixup_v_scale_factor)

        with torch.no_grad():
            # Non transformer init
            if self.unit_emb_dim != 0:
                self.spike_emb.weight.uniform_(-self.initrange, self.initrange)

            self.readout_layer.bias.data.zero_()
            self.readout_layer.weight.data.uniform_(-self.initrange, self.initrange)

    @classmethod
    def create_search_space(cls, trial: optuna.Trial, cfg: DictConfig):
        # model
        unit_emb_dim = trial.suggest_int("model.unit_emb_dim", 0, 2, step=1)
        if unit_emb_dim == 2:
            trial.suggest_int("encoder_num_heads_log2", 0, 1, step=1)
        trial.suggest_int("encoder_ffn_factor_log2", 0, 1, step=1)
        trial.suggest_int("model.encoder_num_layers", 1, 6, step=1)
        trial.suggest_float("dropout", 0.2, 0.6)
        trial.suggest_categorical("model.use_custom_init", [True, False])

        # masking
        trial.suggest_float("masker.mask_ratio", 0.5, 0.9)
        trial.suggest_int("masker.max_block_size", 1, 7, step=1)
        trial.suggest_float("masker.block_mask_prob", 0.0, 1.0)

        # training
        trial.suggest_int("num_epochs", 100, 500, step=200)
        trial.suggest_int("batch_size_log2", 4, 6, step=1)
        trial.suggest_float("weight_decay", 1e-6, 1e-1, log=True)

        # lr scheduler
        trial.suggest_float("base_lr", 1e-5, 5e-2, log=True)
        trial.suggest_float("pct_start", 0.1, 0.9)
        trial.suggest_float("div_factor", 1, 5, log=True)

    @classmethod
    def process_tunable_params(cls, tune_params: dict) -> dict:
        if "encoder_num_heads_log2" in tune_params:
            encoder_num_heads_log2 = tune_params.pop("encoder_num_heads_log2")
            encoder_num_heads = 2**encoder_num_heads_log2
            tune_params["model.encoder_num_heads"] = encoder_num_heads

        encoder_ffn_factor_log2 = tune_params.pop("encoder_ffn_factor_log2")
        encoder_ffn_factor = 2**encoder_ffn_factor_log2
        tune_params["model.encoder_ffn_factor"] = encoder_ffn_factor

        batch_size_log2 = tune_params.pop("batch_size_log2")
        batch_size = 2**batch_size_log2
        tune_params["batch_size"] = batch_size

        dropout = tune_params.pop("dropout")
        tune_params["model.encoder_dropout"] = dropout
        tune_params["model.pre_encoder_dropout"] = dropout
        tune_params["model.post_encoder_dropout"] = dropout

        return tune_params
