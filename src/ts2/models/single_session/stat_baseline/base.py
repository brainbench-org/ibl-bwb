"""Shared skeleton for the training-free TS2 statistical baselines.

A method supplies ``fit`` (closed-form parameters on train windows, hyperparameters
selected on val) and ``predict`` (a closed-form transform of the observed spikes).
Nothing is trained; ``forward`` never builds a gradient path. See
``StatBaselineTrainer`` for how these run without an optimizer.

The two tasks hold out along orthogonal axes, and each method applies to exactly
one: ``co_smoothing`` holds out units, leaving the rest of the population observed
simultaneously; ``forecasting`` holds out trailing timesteps, leaving only each
unit's own past. A subclass declares its axis via ``TASK``, and on the other task
it has no input to work with, so ``link_datasets`` marks it degenerate and
``forward`` returns the per-unit mean rate.
"""

import logging

import numpy as np
import torch
from numpy.lib.stride_tricks import sliding_window_view
from torch_brain.data import Data
from torch_brain.samplers import SequentialFixedWindowSampler
from torch_brain.utils.binning import bin_spikes

from core.model import BaseModel
from ibl_bwb_eval.metrics import PoissonD2Score
from ts2.ts2_dataset import IBLBrainWideBenchTS2

LOG_EPS = 1e-6  # floor for log() of a rate / population multiplier
BIN_EPS = 1e-3  # bin_spikes' rounding tolerance when counting bins

log = logging.getLogger(__name__)


class StatBaseline(BaseModel):
    """Shared skeleton for every statistical baseline.

    Fits the per-unit mean rate and serves it as the default prediction, so a
    method that does not apply to the active task degenerates here instead of
    carrying its own fallback. ``MeanRate`` is that behaviour as a baseline
    in its own right; every other method overrides ``fit`` and ``predict``.
    Outputs log-rates ``(B, T, N)`` for the Poisson NLL loss.

    Not tested at any ``context_length`` other than the default (1.0); see
    :ref:`dataset_context_length`.
    """

    TASK: str | None = None  # task this method applies to; None means any
    NAME: str = "stat_baseline"  # short name, used in log lines

    def __init__(self):
        super().__init__()
        # (N,) log per-unit mean count per bin. A buffer so it moves with
        # .to(device) and lands in the checkpoint.
        self.register_buffer("log_mean", torch.zeros(0), persistent=True)

        self.bin_size: float | None = None
        self.T: int | None = None
        self.N: int | None = None
        self.forecast_ind: int | None = None
        self.task: str | None = None
        self.fit_step_bins: int | None = None
        self.val_score: float | None = None
        self._degenerate: bool = False
        self._window_cache: dict[str, np.ndarray] = {}

    def input_fn(self, data: Data) -> dict:
        """Attached to the dataset transform pipeline by the trainer.

        The target is built by ``IBLBrainWideBenchTS2._get_target``, not here.
        """
        return {"model_inputs": {"spikes": self._bin(data)}}

    def _bin(self, data: Data) -> np.ndarray:
        return bin_spikes(
            data.spikes,
            num_units=len(data.units),
            bin_size=self.bin_size,
            dtype=np.float32,
        )

    def link_datasets(
        self,
        train_dataset: IBLBrainWideBenchTS2,
        val_dataset: IBLBrainWideBenchTS2,
        test_dataset: IBLBrainWideBenchTS2 | None = None,
    ):
        self.bin_size = train_dataset.BIN_SIZE
        self.N = len(train_dataset.get_unit_ids())
        self.T = round(train_dataset.context_window / train_dataset.BIN_SIZE)
        self.task = train_dataset.task
        # Fixed regardless of context_window
        self.forecast_ind = round(
            train_dataset.FORECAST_RATIO * train_dataset.TARGET_WINDOW / train_dataset.BIN_SIZE
        )
        self.fit_step_bins = self.forecast_ind if self.task == "forecasting" else None

        # mean_i = (1 / (n_windows * T)) * sum_windows sum_t s_i(t), train only
        mean = self._windows(train_dataset).reshape(-1, self.N).mean(axis=0)  # (N,)
        log_mean = np.log(np.clip(mean, LOG_EPS, None)).astype(np.float32)
        self.log_mean = torch.from_numpy(log_mean)

        # reported as val/*, until fit overwrites it with its own selection score
        self.val_score = self._mean_rate_val_score(val_dataset)

        self._degenerate = self.TASK is not None and self.task != self.TASK
        if self._degenerate:
            log.warning(
                f"{self.NAME}: task={self.task} is not {self.TASK}, so this method has "
                f"no input to work with; falling back to the per-unit mean rate."
            )
            self._window_cache.clear()
            return

        self.fit(train_dataset, val_dataset)
        self._window_cache.clear()

    # Subclasses override these two.
    def fit(
        self,
        train_dataset: IBLBrainWideBenchTS2,
        val_dataset: IBLBrainWideBenchTS2,
    ):
        """Fit on train, select hyperparameters on val, and record ``val_score``.

        Subclasses pull what they need via :meth:`_windows` or off the dataset. The mean
        rate needs neither, so this is a no-op and the default ``val_score`` already
        covers it (see ``MeanRate``).
        """

    def predict(self, spikes: torch.Tensor, **model_inputs) -> torch.Tensor:
        """Log-rates (B, T, N).

        At test the held-out entries are already zeroed, so the observed population /
        past is exactly the non-masked data.
        """
        return self._log_mean_like(spikes)

    def forward(self, spikes: torch.Tensor, **model_inputs) -> torch.Tensor:
        spikes = spikes.nan_to_num(0.0)
        if self._degenerate:
            return self._log_mean_like(spikes)
        return self.predict(spikes, **model_inputs)

    def _log_mean_view(self, spikes: torch.Tensor) -> torch.Tensor:
        """Per-unit log mean, (1, 1, N) broadcastable view."""
        return self.log_mean.to(spikes.device).view(1, 1, spikes.shape[-1])

    def _log_mean_like(self, spikes: torch.Tensor) -> torch.Tensor:
        """Per-unit log mean, expanded to (B, T, N) and writable."""
        B, T, N = spikes.shape
        return self._log_mean_view(spikes).expand(B, T, N).contiguous()

    @property
    def _obs_end(self) -> int:
        """First forecast bin index, i.e. the end of the observed portion."""
        return self.T - self.forecast_ind

    def _check_horizon(self):
        assert self._obs_end >= 1 and self.forecast_ind >= 1, (
            f"{self.NAME} needs both an observed and a forecast portion "
            f"(T={self.T}, forecast_ind={self.forecast_ind})"
        )

    @staticmethod
    def _score(pred_log_rates: torch.Tensor, target_counts: torch.Tensor) -> float:
        """Mean ``poisson_d2`` over all units and windows, the metric val and test report."""
        metric = PoissonD2Score()
        B, T, N = pred_log_rates.shape
        metric.update(pred_log_rates.reshape(B * T, N), target_counts.reshape(B * T, N))
        return float(metric.compute())

    def _mean_rate_val_score(self, val_dataset: IBLBrainWideBenchTS2) -> float:
        """Val score of the mean rate: what MeanRate and a degenerate method serve.

        Scored over the entries the task holds out, the forecast tail or the val
        hold-out units, so it is comparable to the same method's test score.
        """
        counts = torch.from_numpy(self._windows(val_dataset))  # (B, T, N)
        pred = self._log_mean_like(counts)
        if self.task == "forecasting":
            return self._score(pred[:, self._obs_end :], counts[:, self._obs_end :])
        attr = f"ts2_co_smoothing_is_held_out_{val_dataset.split}"
        units = val_dataset.get_recording(val_dataset.recording_id).units
        held = torch.from_numpy(np.where(np.asarray(getattr(units, attr), dtype=bool))[0])
        return self._score(pred.index_select(2, held), counts.index_select(2, held))

    def _windows(self, dataset: IBLBrainWideBenchTS2) -> np.ndarray:
        """Cached context windows for a split, keyed by split.

        Walking a split is the dominant cost of a run and several methods want the same
        one twice.
        """
        split = dataset.split
        if split not in self._window_cache:
            self._window_cache[split] = (
                self._collect_windows(dataset)
                if self.fit_step_bins is None
                else self._collect_strided_windows(dataset, self.fit_step_bins)
            )
        return self._window_cache[split]

    def _collect_strided_windows(self, dataset: IBLBrainWideBenchTS2, step_bins: int) -> np.ndarray:
        """Context windows at a stride of ``step_bins``, stacked (n_windows, T, N).

        One recording read per trial rather than per window: each trial is binned once
        and unfolded. Exact because the stride is a whole number of bins, so the windows
        land on the same bin edges a per-window binning would give, but only once the
        trial is snapped to whole bins: ``bin_spikes`` drops a partial bin from the left,
        which would shift the whole trial's grid off the per-window one.
        """
        rid = dataset.recording_id
        intervals = dataset.get_sampling_intervals()[rid]
        windows = []
        for start, end in zip(intervals.start, intervals.end, strict=True):
            n_bins = int((end - start) / self.bin_size + BIN_EPS)
            if n_bins < self.T:
                continue
            stop = float(start) + n_bins * self.bin_size
            data = dataset.get_recording(rid).slice(float(start), stop)
            counts = self._bin(dataset.dataset_transform(data))  # (n_bins, N)
            view = sliding_window_view(counts, self.T, axis=0)  # (n_bins - T + 1, N, T)
            windows.append(view[::step_bins].transpose(0, 2, 1))  # (n, T, N)
        assert windows, f"no complete windows found while fitting {self.NAME}"
        return np.concatenate(windows, axis=0)

    def _collect_windows(self, dataset: IBLBrainWideBenchTS2) -> np.ndarray:
        """Non-overlapping context windows as stacked counts (n_windows, T, N).

        Unmasked ground truth on every split: only the model *input* is masked on
        val/test, and that input is discarded here, so a fit sees complete windows
        either way. Bins the recording directly rather than going through
        ``dataset[index]``, since ``__getitem__`` extends each index backward for
        its own context handling, which would break the non-overlapping tiling here.
        """
        rid = dataset.recording_id
        sampler = SequentialFixedWindowSampler(
            sampling_intervals=dataset.get_sampling_intervals(),
            window_length=dataset.context_window,
            step=dataset.context_window,  # non-overlapping
        )
        windows: list[np.ndarray] = []
        for index in sampler:
            data = dataset.get_recording(rid).slice(float(index.start), float(index.end))
            counts = self._bin(dataset.dataset_transform(data))  # (T, N) np.float32
            if counts.shape[0] == self.T:  # drop any short trailing window
                windows.append(counts)
        assert windows, "no complete windows found while fitting StatBaseline"
        return np.stack(windows, axis=0)  # (n_windows, T, N)
