import math
from typing import Literal

import numpy as np
import optuna
import torch
import torch.nn as nn
from omegaconf import DictConfig
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from torch_brain.utils.binning import bin_spikes

from core.dataset import IBLBrainWideBench2026
from core.model import BaseModel
from core.nn import Embedding, tfixup_init_
from ibl_bwb_eval.tasks import ReadoutSpec, TargetResolution


class NDTSuperv(BaseModel):
    """Single-session Transformer encoder for supervised neural decoding :cite:`ndt`.

    Reference implementation: `neural-data-transformers
    <https://github.com/snel-repo/neural-data-transformers>`_.

    .. note::
        Only single-session inference is supported.

    Notation: :math:`B` = batch size, :math:`T_{in}` = input time bins, :math:`N` = units,
    :math:`D_{out}` = task output dim, :math:`D` = hidden dim.

    :meth:`link_datasets` must be called first to build the spike embedding,
    positional embedding, and Transformer encoder (their sizes depend on
    :math:`N`).

    :meth:`configure_readout` must then be called to fix
    :math:`D_{out}` and the output shape.

    1. :meth:`input_fn`: bin raw spikes into :math:`(T_{in}, N)` and generate
       position indices.
    2. :meth:`forward`: embed :math:`(B, T_{in}, N)` to :math:`(B, T_{in}, D)`,
       add positional embeddings, encode with a Transformer, and project to
       :math:`(B, 1, D_{out})` or :math:`(B, T_{in}, D_{out})` depending on the target resolution.

    Not tested at any ``context_length`` other than the default (1.0); see
    :ref:`dataset_context_length`.

    Args:
        bin_size: Width of each time bin in seconds.
        max_spikes: Maximum spike count per bin (used by the count embedding).
        spike_readin: How binned spikes enter the model (see :meth:`link_datasets`).
        hidden_dim: Transformer hidden dimension :math:`D` (overridden by
            ``spike_readin`` ``"none"`` or ``"count"``).
        encoder_num_heads: Number of attention heads.
        encoder_ffn_factor: Feed-forward sublayer width multiplier relative to :math:`D`.
        encoder_dropout: Dropout inside each encoder layer.
        encoder_activation: Activation of the feed-forward sublayer.
        encoder_num_layers: Number of Transformer encoder layers.
        pre_encoder_dropout: Dropout applied to embeddings before the encoder.
        post_encoder_dropout: Dropout applied to encoder output.
        use_custom_init: If ``True``, apply T-Fixup weight initialisation (see :meth:`custom_init`).
        initrange: Uniform init range for non-Transformer weights.
        tfixup_scale_base: T-Fixup base scale factor (required when
            ``use_custom_init=True``).
        tfixup_v_scale_factor: Additional scale applied to value weights
            (required when ``use_custom_init=True``).
    """

    def __init__(
        self,
        bin_size: float,
        max_spikes: int,
        spike_readin: Literal["none", "count", "linear"],
        hidden_dim: int,
        encoder_num_heads: int,
        encoder_ffn_factor: int,
        encoder_dropout: float,
        encoder_activation: str,
        encoder_num_layers: int,
        pre_encoder_dropout: float,
        post_encoder_dropout: float,
        use_custom_init: bool = False,
        initrange: float = 0.1,
        tfixup_scale_base: float | None = None,
        tfixup_v_scale_factor: float | None = None,
    ):
        super().__init__()
        self.bin_size = bin_size
        self.max_spikes = max_spikes
        self.spike_readin = spike_readin
        self.hidden_dim = hidden_dim

        # postpone the init of the encoder to link_datasets
        # as we do not know yet the number of units
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
        train_dataset: IBLBrainWideBench2026,
        val_dataset: IBLBrainWideBench2026,
        test_dataset: IBLBrainWideBench2026 | None = None,
    ) -> None:
        r"""Build session-specific components from the training dataset.

        Must be called before :meth:`configure_readout`. Reads :math:`N` from
        the dataset and builds:

        - **Spike readin**: maps :math:`(B, T_{in}, N)` to
          :math:`(B, T_{in}, D)`, where :math:`D` depends on
          ``spike_readin``:

          - ``"none"``: identity cast; forces :math:`D = N`.
          - ``"count"``: per-unit count lookup of size 2, concatenated;
            forces :math:`D = 2N`.
          - ``"linear"``: linear projection; keeps the configured :math:`D`.

        - **Positional embedding**: lookup table of shape
          :math:`(T_{in}, D)`.
        - **Transformer encoder**: :math:`d_{model} = D`,
          :math:`d_{ff} = D \cdot encoder\_ffn\_factor`.

        Args:
            train_dataset: Training dataset; determines :math:`N` and the
                context window length.
            val_dataset: Validation dataset.
            test_dataset: Test dataset.
        """
        num_unit_per_recording = train_dataset.get_num_unit_per_recording()
        assert len(num_unit_per_recording) == 1, (
            "NDT is single session model, use NDT-stitch for multi sessions"
        )

        self.num_units = next(iter(num_unit_per_recording.values()))

        # Named apart on purpose: `spike_emb` is a lookup table and `spike_proj` a weight
        # matrix, and the two want opposite weight decay (see `no_weight_decay`).
        self.spike_emb = None
        self.spike_proj = None
        if self.spike_readin == "none":
            # Raw spike counts passed directly; hidden_dim = num_units, nhead forced to 1
            self.hidden_dim = self.num_units
            self.encoder_num_heads = 1
        elif self.spike_readin == "count":
            # Per-unit spike-count embedding of dim 2; hidden_dim = num_units * 2, nhead forced to 2
            self.hidden_dim = self.num_units * 2
            self.encoder_num_heads = 2
            self.spike_emb = Embedding(self.max_spikes, 2)
        elif self.spike_readin == "linear":
            # Linear projection from num_units to hidden_dim
            self.spike_proj = nn.Linear(self.num_units, self.hidden_dim)
        else:
            raise ValueError(f"Unknown spike_readin: {self.spike_readin!r}")

        self.emb_scale = math.sqrt(self.hidden_dim)

        ctx_window = train_dataset.context_window
        self.num_bins = int(ctx_window / self.bin_size)
        self.position_emb = Embedding(self.num_bins, self.hidden_dim)

        encoder_layer = TransformerEncoderLayer(
            d_model=self.hidden_dim,
            nhead=self.encoder_num_heads,
            dim_feedforward=self.hidden_dim * self.encoder_ffn_factor,
            dropout=self.encoder_dropout,
            activation=self.encoder_activation,
            norm_first=True,
            batch_first=True,
        )

        # pre-norm layers disable the nested tensor fast path
        self.encoder = TransformerEncoder(
            encoder_layer=encoder_layer,
            num_layers=self.encoder_num_layers,
            norm=nn.LayerNorm(self.hidden_dim),
            enable_nested_tensor=False,
        )

    def configure_readout(self, readout_spec: ReadoutSpec):
        r"""Fix :math:`D_{out}` and build the linear readout head.

        The output shape depends on the target resolution:

        - **Sequence-level** (:math:`T_{out}=1`): project :math:`D \to D_{out}` at
          each time step, then mean-pool to :math:`(B, 1, D_{out})`.
        - **Timestep-level**: project :math:`D \to D_{out}` at each time step,
          output :math:`(B, T_{in}, D_{out})`.

        Args:
            readout_spec: Task specification carrying :math:`D_{out}` and the target resolution.
        """
        output_dim = readout_spec.dim
        self.should_pool = readout_spec.target_resolution == TargetResolution.SEQUENCE

        self.readout = nn.Linear(self.hidden_dim, output_dim)

        if self.use_custom_init:
            self.custom_init()

    def input_fn(self, data) -> dict[str, torch.Tensor]:
        """Bin spikes and generate position indices.

        Args:
            data: Trial data containing raw spike times and unit metadata.

        Returns:
            Dict with:

            - ``model_inputs.spikes``: :math:`(T_{in}, N)` int spike counts.
            - ``model_inputs.positions``: :math:`(T_{in},)` int position indices.
        """
        binned_spikes = bin_spikes(
            spikes=data.spikes,
            num_units=len(data.units),
            bin_size=self.bin_size,
            max_spikes=self.max_spikes,
            dtype=np.int64,
        )  # (T, N)

        position = np.arange(self.num_bins, dtype=np.int64)  # (T,)

        return {
            "model_inputs": {
                "spikes": binned_spikes,
                "positions": position,
            }
        }

    def forward(self, spikes: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        """Map binned spikes to task predictions.

        Args:
            spikes: :math:`(B, T_{in}, N)` int binned spike counts.
            positions: :math:`(T_{in},)` int position indices.

        Returns:
            :math:`(B, 1, D_{out})` for sequence-level tasks or
            :math:`(B, T_{in}, D_{out})` for timestep-level tasks.
        """
        if self.spike_readin == "none":
            spike_repr = spikes.float()
        elif self.spike_readin == "count":
            spike_repr = self.spike_emb(spikes)
            spike_repr = spike_repr.flatten(-2)  # (B, T, N, D) -> (B, T, N*D)
        elif self.spike_readin == "linear":
            spike_repr = self.spike_proj(spikes.float())

        spike_repr = spike_repr * self.emb_scale

        emb = spike_repr + self.position_emb(positions)
        emb = self.pre_encoder_dropout(emb)

        latent = self.encoder(emb)
        latent = self.post_encoder_dropout(latent)

        out = self.readout(latent)  # (B, T, D) -> (B, T, D_out)
        if self.should_pool:
            out = torch.mean(out, dim=1, keepdim=True)  # (B, T, D_out) -> (B, 1, D_out)

        return out

    def custom_init(self):
        """Apply the same weight initialisation scheme used in NDT :cite:`ndt`.

        Combines the T-Fixup strategy :cite:`tfixup` with the
        stabilisation technique from :cite:`no_tears` to scale
        Transformer weights for better optimisation without a warm-up schedule.
        """
        tfixup_init_(self.encoder, self.tfixup_scale_base, self.tfixup_v_scale_factor)

        with torch.no_grad():
            # Non transformer init
            if self.spike_readin == "count":
                self.spike_emb.weight.data.uniform_(-self.initrange, self.initrange)
            elif self.spike_readin == "linear":
                self.spike_proj.weight.data.uniform_(-self.initrange, self.initrange)
                self.spike_proj.bias.data.zero_()

            self.readout.bias.data.zero_()
            self.readout.weight.data.uniform_(-self.initrange, self.initrange)

    @classmethod
    def create_search_space(cls, trial: optuna.Trial, cfg: DictConfig):
        # model
        trial.suggest_int("hidden_dim_log2", 5, 8, step=1)
        trial.suggest_int("encoder_num_heads_log2", 0, 2, step=1)
        trial.suggest_int("model.encoder_num_layers", 2, 10, step=2)
        trial.suggest_int("encoder_ffn_factor_log2", 0, 2, step=1)
        trial.suggest_float("dropout", 0.2, 0.6, step=0.2)
        trial.suggest_categorical("model.use_custom_init", [True, False])

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
        hidden_dim_log2 = tune_params.pop("hidden_dim_log2")
        hidden_dim = 2**hidden_dim_log2
        tune_params["model.hidden_dim"] = hidden_dim

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
