import numpy as np
import optuna
import torch
import torch.nn as nn
from omegaconf import DictConfig
from torch_brain.data import Data
from torch_brain.utils.binning import bin_spikes

from core.model import BaseModel
from ibl_bwb_eval.tasks import ReadoutSpec


class GRU(BaseModel):
    """Gated recurrent unit network mapping binned spike counts to task outputs.

    Notation: :math:`B` = batch size, :math:`T_{in}` = input time bins, :math:`N` = units,
    :math:`D_{out}` = task output dim, :math:`T_{out}` = output time steps,
    :math:`D` = hidden dim (:math:`2D` when bidirectional).

    :meth:`configure_readout` must be called before inference; it fixes
    :math:`D_{out}` and the output shape.

    1. :meth:`input_fn`: bin raw spikes into :math:`(T_{in}, N)`.
    2. :meth:`forward`: project :math:`(B, T_{in}, N)` to :math:`(B, T_{in}, D)`,
       run through ``depth`` GRU layers, adaptively pool to
       :math:`(B, T_{out}, D)`, and project to
       :math:`(B, 1, D_{out})` or :math:`(B, T_{out}, D_{out})` depending on the target resolution.

    Args:
        bin_size: Width of each time bin in seconds.
        depth: Number of stacked GRU layers.
        hidden_dim: Hidden state size :math:`D` per GRU layer.
        dropout: Dropout probability between GRU layers (disabled for
            single-layer models).
        bidirectional: If ``True``, use a bidirectional GRU; the readout
            input dim becomes :math:`2D`.
    """

    def __init__(
        self,
        bin_size: float = 0.02,
        depth: int = 2,
        hidden_dim: int = 64,
        dropout: float = 0.2,
        bidirectional: bool = False,
    ):
        super().__init__()
        assert depth >= 1, f"depth must be >= 1, got {depth}"

        self.bin_size = bin_size

        # Keeps external input interface flexible: (B, T, N) with N inferred lazily.
        self.input_proj = nn.LazyLinear(hidden_dim)

        gru_dropout = dropout if depth > 1 else 0.0
        self.backbone = nn.GRU(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=depth,
            dropout=gru_dropout,
            batch_first=True,
            bidirectional=bidirectional,
        )
        self.out_dim = hidden_dim * (2 if bidirectional else 1)

    def configure_readout(self, readout_spec: ReadoutSpec):
        r"""Fix :math:`D_{out}` and build the temporal pool and readout head.

        The output shape depends on the target resolution:

        - **Sequence-level** (:math:`T_{out}=1`): pool to a single time step,
          project :math:`D \to D_{out}`, reshape to :math:`(B, 1, D_{out})`.
        - **Timestep-level**: pool to :math:`T_{out}` behaviour frames,
          project :math:`D \to D_{out}`, reshape to :math:`(B, T_{out}, D_{out})`.

        Args:
            readout_spec: Task specification carrying :math:`D_{out}` and the target resolution.
        """
        self.readout_spec = readout_spec
        self.temporal_pool = nn.AdaptiveAvgPool1d(readout_spec.num_timesteps)
        self.readout = nn.Linear(self.out_dim, readout_spec.dim)

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
        x = self.input_proj(spikes)  # (B, T_in, N) -> (B, T_in, D)
        x, _ = self.backbone(x)  # (B, T_in, D) -> (B, T_in, D)

        x = x.permute(0, 2, 1)  # (B, T_in, D) -> (B, D, T_in)
        x = self.temporal_pool(x)  # (B, D, T_in) -> (B, D, T_out)
        x = x.permute(0, 2, 1)  # (B, D, T_out) -> (B, T_out, D)

        x = self.readout(x)  # (B, T_out, D) -> (B, T_out, D_out)
        return x

    @classmethod
    def create_search_space(cls, trial: optuna.Trial, cfg: DictConfig):
        # model
        bin_size = trial.suggest_categorical("model.bin_size", [0.001, 0.005, 0.01, 0.02, 0.04])
        trial.suggest_int("model.depth", 1, 3, step=1)
        if bin_size == 0.001:
            trial.suggest_int("hidden_dim_log2", 5, 8, step=1)
        else:
            trial.suggest_int("hidden_dim_log2", 5, 9, step=1)
        trial.suggest_float("model.dropout", 0.0, 0.6, step=0.2)
        trial.suggest_categorical("model.bidirectional", [True, False])

        # training
        trial.suggest_int("num_epochs", 100, 500, step=200)
        trial.suggest_int("batch_size_log2", 4, 6, step=1)
        trial.suggest_float("weight_decay", 1e-6, 1e-1, log=True)

        # lr scheduler
        if hasattr(cfg, "task") and cfg.task == "licking_rate":
            # gru tends to be less stable on licking rate with higher lr
            trial.suggest_float("base_lr", 1e-5, 1e-3, log=True)
        else:
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
