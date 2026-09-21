"""Reduced-rank linear decoder.

Ported from the IBL ``neural_decoding`` reference implementation::

    z[b, t, r] = sum_n  spikes[b, t, n] * U_sess[n, r]     # neurons -> R latents
    y[b, d]    = sum_tr z[b, t, r]      * V[r, t, d] + b_sess[d]

The two correlations it exploits are structural here. Across *trials*: ``U`` carries no
time index, so the rank constrains the neuron axis and every trial fits the same ~35k
parameters at N=1000, T=50, D=50, against ~2.5M dense.

Across *sessions*: ``U``/``b`` are per-session and ``V`` is shared, so the
latents-to-behavior map pools over animals. That split is also what makes transfer work
-- ``U_sess`` absorbs unit count, unit identity and rate scale, so a unit-count mismatch
between pretraining and finetuning needs no alignment.
"""

import numpy as np
import optuna
import torch
import torch.nn as nn
from omegaconf import DictConfig
from torch_brain.data import Data
from torch_brain.utils.binning import bin_spikes

from core.dataset import IBLBrainWideBench2026
from core.model import BaseModel
from core.utils.logger import get_cli_logger
from ibl_bwb_eval.tasks import ReadoutSpec


class RRRDecoder(BaseModel):
    """Reduced-rank decoder mapping a window of binned spikes to behavior :cite:`rrr`.

    Args:
        temporal_rank: rank R of the factorization, the model's main regularizer.
            Constrained to ``R <= min(min_N, T)``.
        bin_size: spike bin width in seconds; ``T = context_window / bin_size``.

    Attributes:
        Us: ``(N_sess, R)`` neuron -> latent projections, keyed by session id so a
            checkpoint survives a change in which sessions are included.
        V: ``(R, T, out_dim)`` read-out. Shared, and the only tensor that transfers.
        bs: ``(out_dim,)`` intercepts, keyed by session id.

    Not tested at any ``context_length`` other than the default (1.0); see
    :ref:`dataset_context_length`.
    """

    def __init__(self, temporal_rank: int = 10, bin_size: float = 0.02):
        super().__init__()
        self.temporal_rank = temporal_rank
        self.bin_size = bin_size
        self.logger = get_cli_logger()

        # Built by link_datasets (needs N) then configure_readout (needs out_dim).
        self.Us = nn.ParameterDict()
        self.bs = nn.ParameterDict()
        self.V: nn.Parameter | None = None
        self.T: int | None = None
        self.out_dim: int | None = None
        self.session_ids: list[str] = []
        self.session_to_index: dict[str, int] = {}

    def link_datasets(
        self,
        train_dataset: IBLBrainWideBench2026,
        val_dataset: IBLBrainWideBench2026,
        test_dataset: IBLBrainWideBench2026 | None = None,
    ):
        # Union over splits; one file per session, so counts cannot disagree.
        unit_counts: dict[str, int] = {}
        for ds in (train_dataset, val_dataset):
            unit_counts.update(ds.get_num_unit_per_recording())

        if not unit_counts:
            raise ValueError("no sessions found in the linked datasets")

        self.T = round(train_dataset.context_window / self.bin_size)
        min_n = min(unit_counts.values())
        max_rank = min(min_n, self.T)
        if self.temporal_rank > max_rank:
            raise ValueError(
                f"temporal_rank={self.temporal_rank} exceeds min(min_N, T)={max_rank} "
                f"(smallest session has {min_n} units, T={self.T} bins). The "
                f"factorization would be rank-deficient; lower temporal_rank, lower "
                f"bin_size, or drop the smallest sessions."
            )

        self.session_ids = sorted(unit_counts)
        self.session_to_index = {sid: i for i, sid in enumerate(self.session_ids)}
        self.Us = nn.ParameterDict(
            {
                sid: nn.Parameter(
                    torch.randn(unit_counts[sid], self.temporal_rank) / unit_counts[sid] ** 0.5
                )
                for sid in self.session_ids
            }
        )
        self.logger.info(
            f"RRRDecoder: {len(self.session_ids)} session(s), "
            f"N in [{min_n}, {max(unit_counts.values())}], T={self.T}, R={self.temporal_rank}"
        )

    def configure_readout(self, readout_spec: ReadoutSpec):
        self.readout_spec = readout_spec

        out_dim = readout_spec.dim * readout_spec.num_timesteps
        self.out_dim = out_dim

        self.V = nn.Parameter(
            torch.randn(self.temporal_rank, self.T, out_dim) / (self.temporal_rank * self.T) ** 0.5
        )
        self.bs = nn.ParameterDict(
            {sid: nn.Parameter(torch.zeros(out_dim)) for sid in self.session_ids}
        )

    def load_ckpt(self, ckpt: dict):
        """Transfer the shared basis ``V``.

        ``Us``/``bs`` index the pretraining sessions' neurons and baselines, so they
        keep their fresh initialization and are fit on this session.
        """
        state_dict = ckpt["model_state_dict"]

        if "V" not in state_dict:
            raise KeyError(
                f"Checkpoint has no 'V'; it does not look like a reduced-rank "
                f"checkpoint (keys: {sorted(state_dict)[:8]})"
            )

        want, got = tuple(self.V.shape), tuple(state_dict["V"].shape)
        if want != got:
            raise ValueError(
                f"Cannot transfer 'V': checkpoint has {got}, this run needs {want}. "
                f"V is (temporal_rank, T, out_dim), so temporal_rank, "
                f"context_window/bin_size, and the task's output dim must all match "
                f"the pretraining run."
            )

        with torch.no_grad():
            self.V.copy_(state_dict["V"])
        self.logger.info(
            f"Transferred shared basis V {want} from checkpoint; U and b are fit on this session"
        )

    def input_fn(self, data: Data) -> dict:
        session_id = data.session.id
        if session_id not in self.Us:
            raise KeyError(
                f"session {session_id} has no U; it was not present when link_datasets "
                f"ran ({len(self.session_ids)} sessions known)"
            )
        binned_spikes = bin_spikes(
            data.spikes,
            num_units=len(data.units),
            bin_size=self.bin_size,
            dtype=np.float32,
        )  # (T, N)
        return {
            "model_inputs": {
                "spikes": binned_spikes,
                "session_index": np.int64(self.session_to_index[session_id]),
            }
        }

    def forward(self, spikes: torch.Tensor, session_index: torch.Tensor) -> torch.Tensor:
        # Checked, not trusted: a mixed batch would index the wrong U and still run.
        idx = session_index.reshape(-1)
        if not bool((idx == idx[0]).all()):
            raise ValueError(
                "batch mixes sessions, so no single U applies. Wrap the sampler in "
                "core.samplers.SessionBatchSampler."
            )

        session_id = self.session_ids[int(idx[0])]
        U, b = self.Us[session_id], self.bs[session_id]

        if spikes.shape[1:] != (self.T, U.shape[0]):
            raise ValueError(
                f"expected spikes of shape (B, {self.T}, {U.shape[0]}) for session "
                f"{session_id}, got {tuple(spikes.shape)}"
            )

        # Neurons first: folding U into V costs ~4x the FLOPs for the same result.
        z = torch.einsum("btn,nr->btr", spikes, U)  # (B, T, R)
        out = torch.einsum("btr,rtd->bd", z, self.V) + b  # (B, out_dim)

        return out.view(out.shape[0], -1, self.readout_spec.dim)  # (B, T_out, D)

    @classmethod
    def create_search_space(cls, trial: optuna.Trial, cfg: DictConfig):
        # model
        trial.suggest_categorical("model.temporal_rank", [2, 3, 5, 8, 12, 20])
        trial.suggest_categorical("model.bin_size", [0.005, 0.01, 0.02, 0.04])

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
        tune_params["batch_size"] = 2**batch_size_log2

        return tune_params
