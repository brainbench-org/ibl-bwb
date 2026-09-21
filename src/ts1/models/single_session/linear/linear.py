import numpy as np
import optuna
import torch
import torch.nn as nn
from omegaconf import DictConfig
from torch_brain.data import Data
from torch_brain.utils.binning import bin_spikes

from core.model import BaseModel
from ibl_bwb_eval.tasks import ReadoutSpec


class Linear(BaseModel):
    r"""Single linear layer mapping binned spike counts to task outputs.

    Notation: :math:`B` = batch size, :math:`T_{in}` = input time bins, :math:`N` = units,
    :math:`D_{out}` = task output dim, :math:`T_{out}` = output time steps.

    :meth:`configure_readout` must be called before inference; it fixes
    :math:`D_{out}` and the output shape.

    1. :meth:`input_fn`: bin raw spikes into :math:`(T_{in}, N)` then flatten to
       :math:`(T_{in} \cdot N,)`.
    2. :meth:`forward`: project :math:`(B, T_{in} \cdot N)` and reshape to
       :math:`(B, 1, D_{out})` or :math:`(B, T_{out}, D_{out})` depending on the target resolution.

    Args:
        bin_size: Width of each time bin in seconds.
    """

    def __init__(
        self,
        bin_size: float = 0.02,
    ):
        super().__init__()
        self.bin_size = bin_size

    def configure_readout(self, readout_spec: ReadoutSpec):
        r"""Fix :math:`D_{out}` and build the linear projection head.

        The output shape depends on the target resolution:

        - **Sequence-level** (:math:`T_{out}=1`): linear
          :math:`(T_{in} \cdot N) \to D_{out}`, reshaped to :math:`(B, 1, D_{out})`.
        - **Timestep-level**: :math:`D_{out}` is expanded by :math:`T_{out}`,
          so linear :math:`(T_{in} \cdot N) \to D_{out} \cdot T_{out}`,
          reshaped to :math:`(B, T_{out}, D_{out})`.

        Args:
            readout_spec: Task specification carrying :math:`D_{out}` and the target resolution.
        """
        self.readout_spec = readout_spec
        # dense regression output is flattened over the task's target timesteps
        output_dim = readout_spec.dim * readout_spec.num_timesteps

        self.linear = nn.LazyLinear(output_dim)

    def input_fn(self, data: Data) -> dict[str, torch.Tensor]:
        r"""Bin and flatten spikes.

        Args:
            data: Trial data containing raw spike times and unit metadata.

        Returns:
            Dict with ``model_inputs.spikes`` of shape :math:`(T_{in} \cdot N,)`.
        """
        spikes = data.spikes
        num_units = len(data.units)
        binned_spikes = bin_spikes(spikes, num_units, self.bin_size, dtype=np.float32)  # (T, N)
        result = {
            "model_inputs": {
                "spikes": binned_spikes.flatten(),  # (T, N) -> (T*N,)
            },
        }
        return result

    def forward(self, spikes: torch.Tensor) -> torch.Tensor:
        r"""Map flattened binned spikes to task predictions.

        Args:
            spikes: :math:`(B, T_{in} \cdot N)` flattened binned spike counts.

        Returns:
            :math:`(B, 1, D_{out})` for sequence-level tasks or
            :math:`(B, T_{out}, D_{out})` for timestep-level tasks.
        """
        x = self.linear(spikes)  # (B, T_in*N) -> (B, T_out*D_out)
        return x.view(x.shape[0], -1, self.readout_spec.dim)  # -> (B, T_out, D_out)

    @classmethod
    def create_search_space(cls, trial: optuna.Trial, cfg: DictConfig):
        # model
        trial.suggest_categorical("model.bin_size", [0.001, 0.005, 0.01, 0.02, 0.04])

        # training
        trial.suggest_int("num_epochs", 100, 500, step=200)
        trial.suggest_int("batch_size_log2", 4, 6, step=1)
        trial.suggest_float("weight_decay", 1e-6, 1e-1, log=True)

        # lr scheduler
        trial.suggest_float("base_lr", 1e-5, 1e-2, log=True)
        trial.suggest_float("pct_start", 0.1, 0.9)
        trial.suggest_float("div_factor", 1, 5, log=True)

    @classmethod
    def process_tunable_params(cls, tune_params: dict) -> dict:
        batch_size_log2 = tune_params.pop("batch_size_log2")
        batch_size = 2**batch_size_log2
        tune_params["batch_size"] = batch_size

        return tune_params
