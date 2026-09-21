from typing import ClassVar, Literal

import numpy as np
import optuna
import torch
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import DictConfig
from torch_brain.data import Data
from torch_brain.utils.binning import bin_spikes

from core.model import BaseModel
from ibl_bwb_eval.tasks import ReadoutSpec

Activation = Literal["relu", "identity"]


class _SamePadConv1d(nn.Module):
    """Conv1d with explicit same padding. With stride>1 output length is ceil(T/stride)."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        dilation: int = 1,
        stride: int = 1,
    ):
        super().__init__()
        total_pad = dilation * (kernel_size - 1)
        self.left_pad = total_pad // 2
        self.right_pad = total_pad - self.left_pad
        self.conv = nn.Conv1d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            dilation=dilation,
            stride=stride,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.pad(x, (self.left_pad, self.right_pad))
        return self.conv(x)


class _SamePadLazyConv1d(nn.Module):
    """Lazy Conv1d with explicit same padding. With stride>1 output length is ceil(T/stride)."""

    def __init__(
        self,
        out_channels: int,
        kernel_size: int,
        dilation: int = 1,
        stride: int = 1,
    ):
        super().__init__()
        total_pad = dilation * (kernel_size - 1)
        self.left_pad = total_pad // 2
        self.right_pad = total_pad - self.left_pad
        self.conv = nn.LazyConv1d(
            out_channels=out_channels,
            kernel_size=kernel_size,
            dilation=dilation,
            stride=stride,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.pad(x, (self.left_pad, self.right_pad))
        return self.conv(x)


class TCN(BaseModel):
    r"""Temporal convolutional network mapping binned spike counts to task outputs.

    Notation: :math:`B` = batch size, :math:`T_{in}` = input time bins, :math:`N` = units,
    :math:`D_{out}` = task output dim, :math:`T_{out}` = output time steps,
    :math:`D` = hidden dim (Conv1d channels).

    :meth:`configure_readout` must be called before inference; it fixes
    :math:`D_{out}` and the output shape.

    1. :meth:`input_fn`: bin raw spikes into :math:`(T_{in}, N)`.
    2. :meth:`forward`: rearrange to :math:`(B, N, T_{in})`, run through ``depth``
       Conv1d layers to :math:`(B, D, T_{in})`, adaptively pool to
       :math:`(B, D, T_{out})`, and project to
       :math:`(B, 1, D_{out})` or :math:`(B, T_{out}, D_{out})` depending on the target resolution.

    Args:
        bin_size: Width of each time bin in seconds.
        depth: Number of Conv1d layers.
        hidden_dim: Number of channels in each convolutional layer :math:`D`.
        kernel_size: Convolutional kernel width (must be a positive odd integer).
        dropout: Dropout probability applied after each layer.
        batch_norm: If ``True``, insert :class:`~torch.nn.BatchNorm1d` after
            each convolutional layer.
        dilation_base: Base for exponential dilation; layer :math:`i` has dilation
            :math:`\text{dilation_base}^i`. Set to ``1`` for no dilation.
        stride: Stride for each convolutional layer.
        activation: Pointwise non-linearity.
    """

    _ACTIVATIONS: ClassVar[dict] = {"relu": nn.ReLU, "identity": nn.Identity}

    def __init__(
        self,
        bin_size: float = 0.02,
        depth: int = 3,
        hidden_dim: int = 64,
        kernel_size: int = 5,
        dropout: float = 0.2,
        batch_norm: bool = True,
        dilation_base: int = 1,
        stride: int = 1,
        activation: Activation = "relu",
    ):
        super().__init__()
        if depth < 1:
            raise ValueError(f"depth must be >= 1, got {depth}")
        if kernel_size < 1 or kernel_size % 2 == 0:
            raise ValueError(f"kernel_size must be a positive odd integer, got {kernel_size}")
        if stride < 1:
            raise ValueError(f"stride must be >= 1, got {stride}")
        if activation not in self._ACTIVATIONS:
            raise ValueError(
                f"activation must be one of {list(self._ACTIVATIONS)}, got {activation}"
            )

        self.bin_size = bin_size
        self.hidden_dim = hidden_dim

        layers: list[nn.Module] = []
        for i in range(depth):
            dilation = dilation_base**i
            if i == 0:
                conv = _SamePadLazyConv1d(
                    out_channels=hidden_dim,
                    kernel_size=kernel_size,
                    dilation=dilation,
                    stride=stride,
                )
            else:
                conv = _SamePadConv1d(
                    in_channels=hidden_dim,
                    out_channels=hidden_dim,
                    kernel_size=kernel_size,
                    dilation=dilation,
                    stride=stride,
                )
            layers.append(conv)

            if batch_norm:
                layers.append(nn.BatchNorm1d(hidden_dim))

            layers.append(self._ACTIVATIONS[activation]())

            if dropout > 0.0:
                layers.append(nn.Dropout(dropout))

        self.backbone = nn.Sequential(*layers)

    def configure_readout(self, readout_spec: ReadoutSpec):
        r"""Fix :math:`D_{out}` and build the temporal pool and readout head.

        The output shape depends on the target resolution:

        - **Sequence-level** (:math:`T_{out}=1`) - pool to a single time step,
          project :math:`D \to D_{out}`, reshape to :math:`(B, 1, D_{out})`.
        - **Timestep-level** - pool to :math:`T_{out}` behaviour frames,
          project :math:`D \to D_{out}`, reshape to :math:`(B, T_{out}, D_{out})`.

        Args:
            readout_spec: Task specification carrying :math:`D_{out}` and the target resolution.
        """
        self.readout_spec = readout_spec

        self.temporal_pool = nn.AdaptiveAvgPool1d(readout_spec.num_timesteps)

        self.readout = nn.Linear(self.hidden_dim, readout_spec.dim)

    def input_fn(self, data: Data) -> dict[str, torch.Tensor]:
        """Bin spikes.

        Args:
            data: Trial data containing raw spike times and unit metadata.

        Returns:
            Dict with ``model_inputs.spikes`` of shape :math:`(T_{in}, N)`.
        """
        spikes = data.spikes
        num_units = len(data.units)
        binned_spikes = bin_spikes(spikes, num_units, self.bin_size, dtype=np.float32)  # (T, N)
        return {"model_inputs": {"spikes": binned_spikes}}

    def forward(self, spikes: torch.Tensor) -> torch.Tensor:
        """Map binned spikes to task predictions.

        Args:
            spikes: :math:`(B, T_{in}, N)` binned spike counts.

        Returns:
            :math:`(B, 1, D_{out})` for sequence-level tasks or
            :math:`(B, T_{out}, D_{out})` for timestep-level tasks.
        """
        x = spikes.permute(0, 2, 1)  # (B, T_in, N) -> (B, N, T_in)
        x = self.backbone(x)  # (B, N, T_in) -> (B, D, T_in)
        x = self.temporal_pool(x)  # (B, D, T_in) -> (B, D, T_out)
        x = x.permute(0, 2, 1)  # (B, D, T_out) -> (B, T_out, D)
        x = self.readout(x)  # (B, T_out, D) -> (B, T_out, D_out)
        return x

    @classmethod
    def create_search_space(cls, trial: optuna.Trial, cfg: DictConfig):
        # model
        trial.suggest_categorical("model.bin_size", [0.001, 0.005, 0.01, 0.02, 0.04])
        depth_idx = trial.suggest_int("depth_idx", 0, 5, step=1)
        trial.suggest_int("hidden_dim_log2", 5, 8, step=1)
        trial.suggest_int("model.kernel_size", 3, 7, step=2)
        trial.suggest_float("model.dropout", 0.0, 0.6, step=0.2)
        trial.suggest_categorical("model.batch_norm", [True, False])
        if depth_idx == 0:
            trial.suggest_categorical("model.activation", ["relu", "identity"])

        # training
        trial.suggest_int("num_epochs", 100, 500, step=200)
        trial.suggest_int("batch_size_log2", 4, 6, step=1)
        trial.suggest_float("weight_decay", 1e-6, 1e-1, log=True)

        # lr scheduler
        if hasattr(cfg, "task") and cfg.task == "licking_rate":
            # tcn tends to be less stable on licking rate with higher lr
            trial.suggest_float("base_lr", 1e-5, 1e-3, log=True)
        else:
            trial.suggest_float("base_lr", 1e-5, 1e-2, log=True)
        trial.suggest_float("pct_start", 0.1, 0.9)
        trial.suggest_float("div_factor", 1, 5, log=True)

    @classmethod
    def process_tunable_params(cls, tune_params: dict) -> dict:
        depth_idx = tune_params.pop("depth_idx")
        depth = 1 if depth_idx == 0 else 2 * depth_idx
        tune_params["model.depth"] = depth

        hidden_dim_log2 = tune_params.pop("hidden_dim_log2")
        hidden_dim = 2**hidden_dim_log2
        tune_params["model.hidden_dim"] = hidden_dim

        batch_size_log2 = tune_params.pop("batch_size_log2")
        batch_size = 2**batch_size_log2
        tune_params["batch_size"] = batch_size

        return tune_params
