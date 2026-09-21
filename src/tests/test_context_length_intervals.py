"""``context_length`` adds extra context by extending the sampled window backward,
without changing which samples exist.

``get_sampling_intervals`` (TS1 and TS2) always returns the original TARGET_WINDOW-sized
trial definition, regardless of ``context_length``. The extra context is added later,
per sample, in an overridden ``__getitem__`` that slices further back before handing
off to the base class. So the eval set never changes with context length.

A trial near the start of a recording or split may not have the full context behind it.
Real recordings have no guaranteed lead-in (checked against real eval sessions: 3 of 29
have less than 20s before their first trial, one as low as 4.65s), so the extension is
clipped to whatever's available instead of requiring the full amount.

A clipped sample is then shorter than ``context_window``. TS1's ``_get_target`` finds
the trailing TARGET_WINDOW from the data's actual span (``data.domain.end[-1]``) rather
than a fixed offset. TS2's ``_get_target`` needed no change: its target_bins count and
forecasting mask were already independent of ``context_window``.

Also covers ``select_scored_entries`` (``ts2_test_mixin.py``): it used to read the mask
at time index 0, which is wrong once ``context_window`` adds context before the scored
region. It now reads the last timestep, which is always inside the scored window.
"""

import numpy as np
import torch
from torch_brain.data import Data, Interval
from torch_brain.datasets import DatasetIndex

from core.dataset import IBLBrainWideBench2026
from ts1.ts1_dataset import IBLBrainWideBenchTS1
from ts2.ts2_dataset import IBLBrainWideBenchTS2
from ts2.ts2_test_mixin import select_scored_entries

# ----------------------------------------------------------------------------------
# get_sampling_intervals: unchanged by context_length, for both suites.
# ----------------------------------------------------------------------------------


def _fake_wheel_speed_recording(trial_starts, trial_ends, domain=(-100.0, 100.0)):
    """A minimal recording satisfying exactly what ``get_sampling_intervals`` reads for
    the ``wheel_speed`` task."""
    broad = Interval(start=np.array([domain[0]]), end=np.array([domain[1]]))
    movement_window = Interval(start=np.asarray(trial_starts), end=np.asarray(trial_ends))
    task_aligned = Data(domain=broad, movement_window=movement_window)
    wheel = Data(
        domain=broad, _domain=broad, wheel_speed=np.zeros(10), timestamps=np.linspace(*domain, 10)
    )
    return Data(
        domain=broad,
        task_aligned_intervals=task_aligned,
        wheel=wheel,
        ts1_test_domain=broad,
    )


def _ts1_dataset(context_window, recording):
    ds = object.__new__(IBLBrainWideBenchTS1)
    ds.context_window = context_window
    ds.task = "wheel_speed"
    ds.split = "test"
    ds.recording_id = "fake-rec"
    ds.get_recording = lambda rid: recording
    return ds


def test_ts1_sampling_intervals_are_independent_of_context_length():
    recording = _fake_wheel_speed_recording([0.0, 1.2, 10.0], [1.0, 2.2, 11.0])
    at_default = _ts1_dataset(1.0, recording).get_sampling_intervals()["fake-rec"]
    at_longer = _ts1_dataset(20.0, recording).get_sampling_intervals()["fake-rec"]
    assert list(at_default.start) == list(at_longer.start) == [0.0, 1.2, 10.0]
    assert list(at_default.end) == list(at_longer.end) == [1.0, 2.2, 11.0]


def _fake_ts2_recording(trial_starts, trial_ends, split_domain=(0.0, 100.0)):
    broad = Interval(start=np.array([split_domain[0]]), end=np.array([split_domain[1]]))
    islands = Interval(start=np.asarray(trial_starts), end=np.asarray(trial_ends))
    task_aligned = Data(domain=islands)
    return Data(domain=broad, task_aligned_intervals=task_aligned, ts2_test_domain=broad)


def _ts2_dataset(context_window, recording):
    ds = object.__new__(IBLBrainWideBenchTS2)
    ds.context_window = context_window
    ds.TARGET_WINDOW = 1.0  # co_smoothing's value; either task's is 1.0 here
    ds.split = "test"
    ds.recording_id = "fake-rec"
    ds.get_recording = lambda rid: recording
    return ds


def test_ts2_sampling_intervals_are_independent_of_context_length():
    recording = _fake_ts2_recording([10.0, 11.2], [11.0, 12.2])
    at_default = _ts2_dataset(1.0, recording).get_sampling_intervals()["fake-rec"]
    at_longer = _ts2_dataset(20.0, recording).get_sampling_intervals()["fake-rec"]
    assert list(at_default.start) == list(at_longer.start)
    assert list(at_default.end) == list(at_longer.end)


# ----------------------------------------------------------------------------------
# __getitem__: extends backward, clipped to 0, for both suites. The base Dataset's own
# __getitem__ is stubbed to capture the index it receives rather than actually slicing
# real data.
# ----------------------------------------------------------------------------------


def _capture_getitem_index(monkeypatch):
    captured = {}

    def fake_getitem(self, index):
        captured["index"] = index

    monkeypatch.setattr(IBLBrainWideBench2026, "__getitem__", fake_getitem)
    return captured


def test_ts1_getitem_extends_backward_when_context_is_available(monkeypatch):
    captured = _capture_getitem_index(monkeypatch)
    ds = _ts1_dataset(5.0, recording=None)
    ds[DatasetIndex(recording_id="fake-rec", start=10.0, end=11.0)]
    assert captured["index"].start == 6.0  # 10.0 - (5.0 - 1.0)
    assert captured["index"].end == 11.0


def test_ts1_getitem_clips_to_zero_when_not_enough_room(monkeypatch):
    captured = _capture_getitem_index(monkeypatch)
    ds = _ts1_dataset(5.0, recording=None)
    ds[DatasetIndex(recording_id="fake-rec", start=2.0, end=3.0)]
    assert captured["index"].start == 0.0  # 2.0 - 4.0 = -2.0, clipped
    assert captured["index"].end == 3.0


def test_ts1_getitem_is_a_noop_at_the_default_context_length(monkeypatch):
    captured = _capture_getitem_index(monkeypatch)
    ds = _ts1_dataset(1.0, recording=None)
    ds[DatasetIndex(recording_id="fake-rec", start=10.0, end=11.0)]
    assert captured["index"].start == 10.0
    assert captured["index"].end == 11.0


def test_ts2_getitem_extends_backward_when_context_is_available(monkeypatch):
    captured = _capture_getitem_index(monkeypatch)
    ds = _ts2_dataset(5.0, recording=None)
    ds[DatasetIndex(recording_id="fake-rec", start=10.0, end=11.0)]
    assert captured["index"].start == 6.0
    assert captured["index"].end == 11.0


def test_ts2_getitem_clips_to_zero_when_not_enough_room(monkeypatch):
    captured = _capture_getitem_index(monkeypatch)
    ds = _ts2_dataset(5.0, recording=None)
    ds[DatasetIndex(recording_id="fake-rec", start=2.0, end=3.0)]
    assert captured["index"].start == 0.0
    assert captured["index"].end == 3.0


def test_ts2_getitem_is_a_noop_at_the_default_context_length(monkeypatch):
    captured = _capture_getitem_index(monkeypatch)
    ds = _ts2_dataset(1.0, recording=None)
    ds[DatasetIndex(recording_id="fake-rec", start=10.0, end=11.0)]
    assert captured["index"].start == 10.0
    assert captured["index"].end == 11.0


# ----------------------------------------------------------------------------------
# TS1's _get_target must find the trailing TARGET_WINDOW from the data's actual span,
# not a fixed context_window-derived offset, since a clipped sample is shorter.
# ----------------------------------------------------------------------------------


def _fake_choice_data(total_span: float) -> Data:
    """A ``choice`` sample whose trial occupies the trailing 1.0s of a window spanning
    ``[0, total_span]``, matching what __getitem__ + Data.slice produce whether or not
    the window was clipped."""
    choice_interval = Interval(
        start=np.array([total_span - 1.0]), end=np.array([total_span]), choice=np.array([1])
    )
    task_aligned = Data(
        domain=Interval(start=np.array([0.0]), end=np.array([total_span])), choice=choice_interval
    )
    return Data(
        domain=Interval(start=np.array([0.0]), end=np.array([total_span])),
        task_aligned_intervals=task_aligned,
    )


def _ts1_dataset_for_get_target():
    ds = object.__new__(IBLBrainWideBenchTS1)
    ds.task = "choice"
    ds.split = "train"
    return ds


def test_ts1_get_target_finds_trailing_window_at_full_context():
    target = _ts1_dataset_for_get_target()._get_target(_fake_choice_data(5.0))
    assert list(target["values"]) == [1]


def test_ts1_get_target_finds_trailing_window_when_clipped_shorter_than_context_window():
    """Sample clipped to 3.0s from a requested 5.0s; the trailing 1.0s target must
    still be found correctly from the data's actual span."""
    target = _ts1_dataset_for_get_target()._get_target(_fake_choice_data(3.0))
    assert list(target["values"]) == [1]


def test_ts1_get_target_finds_trailing_window_at_the_default_context_length():
    target = _ts1_dataset_for_get_target()._get_target(_fake_choice_data(1.0))
    assert list(target["values"]) == [1]


# ----------------------------------------------------------------------------------
# select_scored_entries
# ----------------------------------------------------------------------------------


def test_select_scored_entries_finds_the_co_smoothing_mask_beyond_default_context():
    """Once context_window grows past the default, the mask is False everywhere except
    the trailing TARGET_WINDOW. Confirm the held-out units are still found there."""
    T, N, target_bins = 5, 3, 2
    held_out_units = np.array([True, False, True])

    mask = torch.zeros((1, T, N), dtype=torch.bool)
    mask[:, -target_bins:, :] = torch.from_numpy(held_out_units)
    pred = torch.arange(T * N, dtype=torch.float32).reshape(1, T, N)
    target = pred.clone() + 1.0

    pred_flat, target_flat, pred_subset, keep = select_scored_entries(
        pred, target, mask, mask_dim=2
    )

    assert list(keep.numpy()) == [0, 2]  # the two held-out units, by index
    assert pred_subset.shape == (1, T, 2)
    assert pred_flat.numel() == T * 2
    assert torch.equal(target_flat, pred_flat + 1.0)


def test_select_scored_entries_still_works_at_the_default_context_window():
    """context_window == TARGET_WINDOW: the mask is True everywhere, so this must behave
    exactly as it did before the fix."""
    T, N = 2, 3
    held_out_units = np.array([True, False, True])

    mask = torch.zeros((1, T, N), dtype=torch.bool)
    mask[:, :, :] = torch.from_numpy(held_out_units)
    pred = torch.arange(T * N, dtype=torch.float32).reshape(1, T, N)
    target = pred.clone()

    _, _, pred_subset, keep = select_scored_entries(pred, target, mask, mask_dim=2)
    assert list(keep.numpy()) == [0, 2]
    assert pred_subset.shape == (1, T, 2)
