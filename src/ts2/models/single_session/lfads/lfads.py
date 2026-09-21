"""LFADS / AutoLFADS for Task Suite 2 (neural prediction).

Sequential VAE of Pandarinath et al. 2018 with the AutoLFADS regularizers of
Keshtkaran, Sedler et al. 2022, written from the published architecture with
`lfads-torch <https://github.com/arsedler9/lfads-torch>`_ as a behavioral reference.

Two deviations from the reference, both for speed: the encoders use fused
``nn.GRU``, so their states are unclipped and the n-gate follows the PyTorch
convention, while the generator and controller keep the reference's clipped
TF-convention cell where the long rollout makes stability matter; and only
single-session models are supported.
"""

from typing import Literal, TypeAlias

import numpy as np
import optuna
import torch
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import DictConfig
from torch.distributions import Independent, Normal, kl_divergence
from torch.distributions.transforms import AffineTransform
from torch_brain.data import Data
from torch_brain.utils.binning import bin_spikes

from core.model import BaseModel
from ts2.ts2_dataset import IBLBrainWideBenchTS2

# Where the forecasting reconstruction gradient comes from: the held-out tail
# only, or the whole window (the observed part then also shapes the dynamics).
ForecastGradScope: TypeAlias = Literal["tail", "window"]


def _init_variance_scaling_(weight: torch.Tensor, scale_dim: int):
    nn.init.normal_(weight, std=1.0 / np.sqrt(scale_dim))


def _init_linear_(linear: nn.Linear):
    _init_variance_scaling_(linear.weight, linear.in_features)
    if linear.bias is not None:
        nn.init.zeros_(linear.bias)


class ClippedGRUCell(nn.GRUCell):
    """GRU cell with clipped hidden state and the LFADS (TensorFlow) n-gate.

    Unlike ``nn.GRUCell`` the reset gate is applied before the recurrent matmul,
    matching the original LFADS. ``bias_hh`` is frozen at zero, gate biases open.
    """

    def __init__(self, input_size: int, hidden_size: int, clip_value: float = float("inf")):
        super().__init__(input_size, hidden_size, bias=True)
        self.clip_value = clip_value

        if input_size > 0:  # input_size == 0 when the controller is disabled
            _init_variance_scaling_(self.weight_ih, input_size)
        _init_variance_scaling_(self.weight_hh, hidden_size)
        nn.init.ones_(self.bias_ih)
        self.bias_ih.data[-hidden_size:] = 0.0
        nn.init.zeros_(self.bias_hh)
        self.bias_hh.requires_grad = False

    def forward(self, input: torch.Tensor, hidden: torch.Tensor) -> torch.Tensor:
        x_z, x_r, x_n = torch.chunk(input @ self.weight_ih.T + self.bias_ih, chunks=3, dim=1)
        weight_hh_zr, weight_hh_n = torch.split(
            self.weight_hh, [2 * self.hidden_size, self.hidden_size]
        )
        h_z, h_r = torch.chunk(hidden @ weight_hh_zr.T, chunks=2, dim=1)

        z = torch.sigmoid(x_z + h_z)
        r = torch.sigmoid(x_r + h_r)
        n = torch.tanh(x_n + (r * hidden) @ weight_hh_n.T)

        hidden = z * hidden + (1 - z) * n
        return torch.clamp(hidden, -self.clip_value, self.clip_value)


class KernelNormalizedLinear(nn.Linear):
    """Linear layer with unit-L2-norm weight rows.

    Used for the generator-to-factors map so factor scale is set by the generator
    state rather than absorbed into the weights.
    """

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return F.linear(input, F.normalize(self.weight, p=2, dim=1), self.bias)


class GaussianPrior(nn.Module):
    """Diagonal Gaussian prior over g0, learnable mean and fixed variance."""

    def __init__(self, variance: float, shape: int):
        super().__init__()
        self.mean = nn.Parameter(torch.zeros(shape), requires_grad=True)
        self.logvar = nn.Parameter(torch.log(torch.ones(shape) * variance), requires_grad=False)

    def forward(self, post_mean: torch.Tensor, post_std: torch.Tensor) -> torch.Tensor:
        """Analytic KL[q || p], summed over the latent dim and averaged over batch."""
        posterior = Independent(Normal(post_mean, post_std), 1)
        prior = Independent(Normal(self.mean, torch.exp(0.5 * self.logvar)), 1)
        return kl_divergence(posterior, prior).mean()


class AutoregressiveGaussianPrior(nn.Module):
    """AR(1) prior over the inferred inputs u_t, with learnable tau and noise.

    Lets the inferred inputs stay temporally smooth instead of being pushed
    toward white noise, which is what keeps the controller usable at full KL.
    """

    def __init__(self, tau: float, nvar: float, shape: int):
        super().__init__()
        self.logtaus = nn.Parameter(torch.log(torch.ones(shape) * tau))
        self.lognvars = nn.Parameter(torch.log(torch.ones(shape) * nvar))

    def log_prob(self, sample: torch.Tensor) -> torch.Tensor:
        alphas = torch.exp(-1.0 / torch.exp(self.logtaus))
        logpvars = self.lognvars - torch.log(1 - alphas**2)

        # p(u_t | u_{t-1}) = N(alpha * u_{t-1}, nvar), with the first step drawn
        # from the stationary distribution N(0, nvar / (1 - alpha^2)).
        prev_sample = torch.roll(sample, shifts=1, dims=1)
        means = AffineTransform(loc=0, scale=alphas)(prev_sample)
        stddevs = torch.ones_like(means) * torch.exp(0.5 * self.lognvars)
        means[:, 0] = 0.0
        stddevs[:, 0] = torch.exp(0.5 * logpvars)

        return Independent(Normal(means, stddevs), 2).log_prob(sample)

    def forward(self, post_mean: torch.Tensor, post_std: torch.Tensor) -> torch.Tensor:
        """Sampled KL[q || p], summed over (time, latent) and averaged over batch."""
        posterior = Independent(Normal(post_mean, post_std), 2)
        sample = posterior.rsample()
        return (posterior.log_prob(sample) - self.log_prob(sample)).mean()


class LFADS(BaseModel):
    """LFADS for TS2 co-smoothing and forecasting :cite:`lfads,autolfads`.

    Consumes binned spike counts (B, T, N) and returns log firing rates of the
    same shape for the benchmark's Poisson NLL.

    During training the input is corrupted in the shape of the task's hold-out and
    the reconstruction gradient flows only through the corrupted entries (see
    :meth:`_mask_input`). Val and test arrive already stripped by the dataset.

    Args:
        ic_enc_dim: Hidden size per direction of the initial-condition encoder.
        ci_enc_dim: Hidden size per direction of the controller-input encoder.
        ic_dim: Dimension of the initial-condition latent g0.
        gen_dim: Generator hidden size.
        fac_dim: Number of latent factors.
        co_dim: Dimension of the inferred inputs u_t (0 disables the controller).
        con_dim: Controller hidden size.
        ci_lag: Lag on the controller input, so the controller does not see the
            current timestep of the data it must explain.
        dropout: Rate applied to input, encoder output, controller input and
            generator state.
        cd_rate: Coordinated-dropout rate: single input entries hidden from the
            encoder and used for the reconstruction gradient. Co-smoothing only.
        unit_cd_rate: Rate of hiding a unit's whole row for the window, the shape
            the co-smoothing hold-out takes at eval. Co-smoothing only.
        cd_pass_rate: Fraction of visible entries that also pass gradient.
        cell_clip: Clip value for generator and controller states.
        ic_post_var_min: Floor on the initial-condition posterior variance.
        log_rate_clamp: Clamp on output log-rates, keeping the Poisson NLL from
            overflowing early in training.
        ic_prior_variance: Variance of the fixed-variance prior over g0.
        co_prior_tau: Initial AR(1) time constant of the prior over u_t, in bins.
        co_prior_nvar: Initial AR(1) process variance of the prior over u_t.
        forecast_grad_scope: Forecasting only, ignored for co-smoothing. Whether
            the reconstruction gradient comes from the held-out tail ("tail") or
            the whole window ("window").
    """

    def __init__(
        self,
        ic_enc_dim: int = 64,
        ci_enc_dim: int = 64,
        ic_dim: int = 64,
        gen_dim: int = 200,
        fac_dim: int = 40,
        co_dim: int = 4,
        con_dim: int = 64,
        ci_lag: int = 1,
        dropout: float = 0.05,
        cd_rate: float = 0.3,
        unit_cd_rate: float = 0.1,
        cd_pass_rate: float = 0.0,
        cell_clip: float = 5.0,
        ic_post_var_min: float = 1e-4,
        log_rate_clamp: float = 8.0,
        ic_prior_variance: float = 0.1,
        co_prior_tau: float = 10.0,
        co_prior_nvar: float = 0.1,
        forecast_grad_scope: ForecastGradScope = "window",
    ):
        super().__init__()
        self.ic_enc_dim = ic_enc_dim
        self.ci_enc_dim = ci_enc_dim
        self.ic_dim = ic_dim
        self.fac_dim = fac_dim
        self.co_dim = co_dim
        self.con_dim = con_dim
        self.ci_lag = ci_lag
        self.cd_rate = cd_rate
        self.unit_cd_rate = unit_cd_rate
        self.cd_pass_rate = cd_pass_rate
        self.forecast_grad_scope = forecast_grad_scope
        self.ic_post_var_min = ic_post_var_min
        self.log_rate_clamp = log_rate_clamp
        self.use_con = co_dim > 0 and con_dim > 0 and ci_enc_dim > 0

        self.dropout = nn.Dropout(dropout)

        # Encoders and rate readout depend on the number of units, so they are
        # created in link_datasets.
        self.ic_enc: nn.GRU | None = None
        self.ci_enc: nn.GRU | None = None
        self.readout: nn.Linear | None = None

        self.ic_enc_h0 = nn.Parameter(torch.zeros(2, 1, ic_enc_dim))
        self.ic_linear = nn.Linear(2 * ic_enc_dim, 2 * ic_dim)
        _init_linear_(self.ic_linear)

        self.ic_to_g0 = nn.Linear(ic_dim, gen_dim)
        _init_linear_(self.ic_to_g0)

        self.gen_cell = ClippedGRUCell(co_dim, gen_dim, clip_value=cell_clip)
        self.fac_linear = KernelNormalizedLinear(gen_dim, fac_dim, bias=False)
        _init_linear_(self.fac_linear)

        self.ic_prior = GaussianPrior(variance=ic_prior_variance, shape=ic_dim)

        if self.use_con:
            self.ci_enc_h0 = nn.Parameter(torch.zeros(2, 1, ci_enc_dim))
            self.con_cell = ClippedGRUCell(2 * ci_enc_dim + fac_dim, con_dim, clip_value=cell_clip)
            self.co_linear = nn.Linear(con_dim, 2 * co_dim)
            _init_linear_(self.co_linear)
            self.co_prior = AutoregressiveGaussianPrior(
                tau=co_prior_tau, nvar=co_prior_nvar, shape=co_dim
            )

        # Set in link_datasets
        self.bin_size: float | None = None
        self.task: str | None = None
        self.forecast_ratio: float | None = None
        self.N: int | None = None

    # ------------------------------------------------------------------
    # Dataset linking: resolves T, N and the task-matched input corruption
    # ------------------------------------------------------------------

    def link_datasets(
        self,
        train_dataset: IBLBrainWideBenchTS2,
        val_dataset: IBLBrainWideBenchTS2 | None = None,
        test_dataset: IBLBrainWideBenchTS2 | None = None,
    ):
        self.bin_size = train_dataset.BIN_SIZE
        self.task = train_dataset.task
        self.forecast_ratio = train_dataset.FORECAST_RATIO
        self.N = len(train_dataset.get_unit_ids())

        self.ic_enc = nn.GRU(self.N, self.ic_enc_dim, bidirectional=True, batch_first=True)
        if self.use_con:
            self.ci_enc = nn.GRU(self.N, self.ci_enc_dim, bidirectional=True, batch_first=True)

        self.readout = nn.Linear(self.fac_dim, self.N)
        _init_linear_(self.readout)

        # Start each unit at its mean log rate so early training does not spend
        # itself learning the offsets.
        recording = train_dataset.get_recording(train_dataset.recording_id)
        assert hasattr(recording.units, "firing_rate"), (
            "LFADS needs units.firing_rate to initialise its rate bias"
        )
        mean_counts = np.asarray(recording.units.firing_rate) * self.bin_size
        bias = np.log(np.clip(mean_counts, 1e-4, None))
        with torch.no_grad():
            self.readout.bias.copy_(torch.as_tensor(bias, dtype=torch.float32))

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

    def forward(self, spikes: torch.Tensor, return_posterior: bool = False) -> torch.Tensor | dict:
        """Run the sequential VAE.

        Args:
            spikes: (B, T, N) binned spike counts.
            return_posterior: Also return the KL terms, L2 penalties and gradient
                mask. A forward argument rather than a second method so training
                still goes through ``DDP.forward``.

        Returns:
            (B, T, N) log firing rates, or a dict with those under ``log_rates``
            plus the training quantities.
        """
        spikes = spikes.nan_to_num(0.0)
        B, T, _N = spikes.shape

        grad_mask = None
        if return_posterior and self.training:
            spikes, grad_mask = self._mask_input(spikes)

        # Encoders
        data_drop = self.dropout(spikes)
        _, h_n = self.ic_enc(data_drop, self.ic_enc_h0.tile(1, B, 1))
        h_n = torch.cat([h_n[0], h_n[1]], dim=1)  # (B, 2 * ic_enc_dim)

        ic_params = self.ic_linear(self.dropout(h_n))
        ic_mean, ic_logvar = torch.split(ic_params, self.ic_dim, dim=1)
        ic_std = torch.sqrt(torch.exp(ic_logvar) + self.ic_post_var_min)
        ic = self._sample(ic_mean, ic_std)

        gen_state = self.ic_to_g0(self.dropout(ic))
        factor = self.fac_linear(self.dropout(gen_state))

        if self.use_con:
            ci = self._encode_controller_input(data_drop, B, T)
            con_state = torch.zeros(B, self.con_dim, device=spikes.device, dtype=gen_state.dtype)

        # Decoder rollout
        log_rates, co_means, co_stds = [], [], []
        for t in range(T):
            if self.use_con:
                con_input = self.dropout(torch.cat([ci[:, t], factor], dim=1))
                con_state = self.con_cell(con_input, con_state)
                co_params = self.co_linear(con_state)
                co_mean, co_logvar = torch.split(co_params, self.co_dim, dim=1)
                co_std = torch.sqrt(torch.exp(co_logvar))
                gen_input = self._sample(co_mean, co_std)
                co_means.append(co_mean)
                co_stds.append(co_std)
            else:
                gen_input = torch.zeros(B, 0, device=spikes.device, dtype=gen_state.dtype)

            gen_state = self.gen_cell(gen_input, gen_state)
            factor = self.fac_linear(self.dropout(gen_state))
            log_rates.append(self.readout(factor))

        log_rates = torch.stack(log_rates, dim=1).clamp(-self.log_rate_clamp, self.log_rate_clamp)

        if not return_posterior:
            return log_rates

        out = {
            "log_rates": log_rates,
            "grad_mask": grad_mask,
            "kl_ic": self.ic_prior(ic_mean.float(), ic_std.float()),
            "l2_gen": self._recurrent_l2(self.gen_cell),
        }
        if self.use_con:
            out["kl_co"] = self.co_prior(
                torch.stack(co_means, dim=1).float(),
                torch.stack(co_stds, dim=1).float(),
            )
            out["l2_con"] = self._recurrent_l2(self.con_cell)
        else:
            zero = log_rates.new_zeros(())
            out["kl_co"] = zero
            out["l2_con"] = zero

        return out

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _sample(self, mean: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
        """Reparameterized sample while training, posterior mean at eval."""
        if not self.training:
            return mean
        return mean + std * torch.randn_like(std)

    def _encode_controller_input(self, data_drop: torch.Tensor, B: int, T: int) -> torch.Tensor:
        ci, _ = self.ci_enc(data_drop, self.ci_enc_h0.tile(1, B, 1))
        ci_fwd, ci_bwd = torch.split(ci, self.ci_enc_dim, dim=2)
        # Delay each direction by ci_lag so step t is explained from data the
        # controller has not already been handed.
        ci_fwd = F.pad(ci_fwd, (0, 0, self.ci_lag, 0, 0, 0))[:, :T]
        ci_bwd = F.pad(ci_bwd, (0, 0, 0, self.ci_lag, 0, 0))[:, -T:]
        return torch.cat([ci_fwd, ci_bwd], dim=2)

    def _mask_input(self, spikes: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Corrupt the encoder input in the shape of the task's hold-out.

        Co-smoothing hides entries and takes the gradient on those alone:
        ``cd_rate`` scatters single entries (what keeps the model off the identity),
        ``unit_cd_rate`` blanks whole rows as val and test do. Forecasting zeroes
        the trailing bins, gradient scope per ``forecast_grad_scope``.

        Training only, val and test arrive already stripped.

        Returns:
            The corrupted input and a float gradient mask, or ``(spikes, None)``
            if no corruption applies.
        """
        if self.task == "forecasting":
            num_bins = int(self.forecast_ratio * spikes.shape[1])
            if num_bins == 0:
                return spikes, None
            masked = spikes.clone()
            masked[:, -num_bins:] = 0.0
            if self.forecast_grad_scope == "window":
                return masked, None
            grad_mask = torch.zeros_like(spikes)
            grad_mask[:, -num_bins:] = 1.0
            return masked, grad_mask

        if self.cd_rate <= 0.0 and self.unit_cd_rate <= 0.0:
            return spikes, None

        B, _T, N = spikes.shape
        masked = spikes
        hidden = torch.zeros_like(spikes, dtype=torch.bool)

        if self.cd_rate > 0.0:
            hidden = torch.rand_like(spikes) < self.cd_rate
            masked = masked * ~hidden / (1 - self.cd_rate)

        if self.unit_cd_rate > 0.0:
            # Silent for every bin and unrescaled, exactly a held-out row at eval.
            held_units = torch.rand(B, 1, N, device=spikes.device) < self.unit_cd_rate
            masked = masked.masked_fill(held_units, 0.0)
            hidden = hidden | held_units

        grad_mask = torch.logical_or(hidden, torch.rand_like(spikes) < self.cd_pass_rate)
        return masked, grad_mask.to(spikes.dtype)

    @staticmethod
    def _recurrent_l2(cell: ClippedGRUCell) -> torch.Tensor:
        """Mean squared recurrent weight, halved, as in the reference."""
        return 0.5 * cell.weight_hh.pow(2).sum() / cell.weight_hh.numel()

    # ------------------------------------------------------------------
    # Tuning
    # ------------------------------------------------------------------

    @classmethod
    def create_search_space(cls, trial: optuna.Trial, cfg: DictConfig):
        # model capacity
        trial.suggest_categorical("model.gen_dim", [100, 200, 400])
        trial.suggest_categorical("model.fac_dim", [20, 40, 80])
        trial.suggest_categorical("model.co_dim", [1, 2, 4, 8])

        # AutoLFADS searches the regularizers rather than fixing them
        trial.suggest_float("model.dropout", 0.0, 0.6, step=0.05)
        trial.suggest_float("kl_ic_scale", 1e-3, 1e1, log=True)
        trial.suggest_float("kl_co_scale", 1e-3, 1e1, log=True)
        trial.suggest_float("l2_gen_scale", 1e-4, 1e2, log=True)
        trial.suggest_float("l2_con_scale", 1e-4, 1e2, log=True)

        # Only the corruption knobs this task's forward pass reads.
        if cfg.task == "forecasting":
            trial.suggest_categorical("model.forecast_grad_scope", ["tail", "window"])
        else:
            trial.suggest_float("model.cd_rate", 0.0, 0.7, step=0.05)
            trial.suggest_float("model.unit_cd_rate", 0.0, 0.3, step=0.05)

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
        batch_size_log2 = tune_params.pop("batch_size_log2")
        tune_params["batch_size"] = 2**batch_size_log2
        return tune_params
