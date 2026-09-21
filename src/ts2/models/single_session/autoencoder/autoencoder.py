from typing import Literal

import numpy as np
import optuna
import torch
import torch.nn as nn
from omegaconf import DictConfig
from torch_brain.data import Data
from torch_brain.utils.binning import bin_spikes

from core.model import BaseModel
from ts2.ts2_dataset import IBLBrainWideBenchTS2

Activation = Literal["relu", "gelu", "tanh"]


class AutoencoderMLP(BaseModel):
    """Simple Autoencoder with MLP encoder/decoder for TS2 neural prediction.

    Takes (masked) binned spike counts of shape (B, T, N) as input and outputs
    log-rates of shape (B, T, N) for the Poisson NLL reconstruction loss.

    N (number of neurons) is variable per recording; it is resolved lazily in
    `link_datasets` so that a single output layer can be created once the
    dataset is available.

    Not tested at any ``context_length`` other than the default (1.0); see
    :ref:`dataset_context_length`.

    Args:
        encoder_depth: Number of encoder layers. Each halves the width after the first.
        decoder_depth: Number of decoder layers. The first keeps the bottleneck width,
            each later one doubles it.
        hidden_dim: Width of the first encoder layer.
        dropout: Dropout probability applied after each activation.
        activation: Pointwise non-linearity.
        batch_norm: If ``True``, insert :class:`~torch.nn.BatchNorm1d` after each linear layer.
    """

    def __init__(
        self,
        encoder_depth: int = 2,
        decoder_depth: int = 2,
        hidden_dim: int = 256,
        dropout: float = 0.2,
        activation: Activation = "relu",
        batch_norm: bool = True,
    ):
        super().__init__()
        model_layers = []

        # encoder
        in_dim = -1
        dim = hidden_dim
        for _ in range(encoder_depth):
            if in_dim == -1:
                model_layers.append(nn.LazyLinear(dim))
            else:
                model_layers.append(nn.Linear(in_dim, dim))
            if batch_norm:
                model_layers.append(nn.BatchNorm1d(dim))
            model_layers.append(self._get_activation(activation))
            if dropout > 0.0:
                model_layers.append(nn.Dropout(dropout))
            in_dim = dim
            dim = dim // 2

        # decoder
        dim = in_dim
        for _ in range(decoder_depth):
            model_layers.append(nn.Linear(in_dim, dim))
            if batch_norm:
                model_layers.append(nn.BatchNorm1d(dim))
            model_layers.append(self._get_activation(activation))
            if dropout > 0.0:
                model_layers.append(nn.Dropout(dropout))
            in_dim = dim
            dim = dim * 2

        self.net = nn.Sequential(*model_layers)
        self.final_hidden_dim = in_dim

        # Readout is created in link_datasets once T and N are known
        self.readout: nn.Linear | None = None

    # ------------------------------------------------------------------
    # Dataset linking: resolves T and N so the output layer can be built
    # ------------------------------------------------------------------

    def link_datasets(
        self,
        train_dataset: IBLBrainWideBenchTS2,
        val_dataset: IBLBrainWideBenchTS2,
        test_dataset: IBLBrainWideBenchTS2 | None = None,
    ):
        self.bin_size = train_dataset.BIN_SIZE
        N = len(train_dataset.get_unit_ids())
        T = round(train_dataset.context_window / train_dataset.BIN_SIZE)
        self.readout = nn.Linear(self.final_hidden_dim, T * N)

    # ------------------------------------------------------------------
    # Input fn: added to the dataset transform pipeline by the trainer
    # ------------------------------------------------------------------

    def input_fn(self, data: Data) -> dict:
        bin_counts = bin_spikes(
            data.spikes,
            num_units=len(data.units),
            bin_size=self.bin_size,
            dtype=np.float32,
        )

        return {
            "model_inputs": {
                "spikes": bin_counts,  # (T, N)
            },
        }

    # ------------------------------------------------------------------
    # Forward pass
    # ------------------------------------------------------------------

    def forward(self, spikes: torch.Tensor) -> torch.Tensor:
        """Predict log spike rates from masked spike counts.

        Args:
            spikes: (B, T, N) masked binned spike counts

        Returns:
            log_rates: (B, T, N) predicted log spike rates
        """
        spikes = spikes.nan_to_num(0.0)
        B, T, N = spikes.shape

        x = spikes.reshape(B, T * N)  # (B, T*N)
        x = self.net(x)  # (B, hidden_dim)
        x = self.readout(x)  # (B, T*N)
        x = x.reshape(B, T, N)  # (B, T, N)

        return x

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _get_activation(self, activation: Activation) -> nn.Module:
        if activation == "relu":
            return nn.ReLU()
        elif activation == "gelu":
            return nn.GELU()
        elif activation == "tanh":
            return nn.Tanh()
        raise ValueError(f"Unknown activation: {activation}")

    @classmethod
    def create_search_space(cls, trial: optuna.Trial, cfg: DictConfig):
        # model
        trial.suggest_int("model.encoder_depth", 1, 3, step=1)
        trial.suggest_int("model.decoder_depth", 1, 3, step=1)
        trial.suggest_int("hidden_dim_log2", 5, 9, step=1)
        trial.suggest_float("model.dropout", 0.0, 0.6, step=0.2)
        trial.suggest_categorical("model.batch_norm", [True, False])

        # training
        trial.suggest_int("num_epochs", 100, 500, step=200)
        trial.suggest_int("batch_size_log2", 4, 7, step=1)
        trial.suggest_float("weight_decay", 1e-6, 1e-1, log=True)

        # lr scheduler
        trial.suggest_float("base_lr", 1e-5, 1e-2, log=True)
        trial.suggest_float("pct_start", 0.1, 0.9)
        trial.suggest_float("div_factor", 1, 5, log=True)

    @classmethod
    def process_tunable_params(cls, tune_params: dict) -> dict:
        hidden_dim_log2 = tune_params.pop("hidden_dim_log2")
        hidden_dim = 2**hidden_dim_log2
        tune_params["model.hidden_dim"] = hidden_dim

        batch_size_log2 = tune_params.pop("batch_size_log2")
        batch_size = 2**batch_size_log2
        tune_params["batch_size"] = batch_size

        return tune_params
