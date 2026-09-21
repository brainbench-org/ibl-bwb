"""Every TS1/TS2 single-session model reads ``train_dataset.context_window`` inside
``link_datasets``, called only once the dataset (and so its ``context_length``) is
known, never in ``__init__``. That ordering is what makes a model safe to point at a
longer context: a fixed-size layer built from a hardcoded ``T`` would silently mismatch
the sampler's actual window once ``context_length`` moved off its 1.0 default.

These tests lock that in as a regression: build each model with a fake dataset at two
``context_window`` values and check the resulting layer scales with it. The fake
dataset is a minimal stand-in for whatever that model's ``link_datasets`` calls, not a
real ``IBLBrainWideBench2026`` (no data build exists in this environment, and none of
this needs one, since every quantity these models derive from the dataset is a plain
Python value, not a tensor read from disk).

``context_length`` is only configurable for TS1/TS2 eval; pretrain models are out of
scope for this pass and untested here.
"""

import numpy as np
import pytest

from ts1.models.single_session.ndt_superv.ndt_superv import NDTSuperv
from ts2.models.single_session.autoencoder.autoencoder import AutoencoderMLP
from ts2.models.single_session.ndt.ndt import NDT
from ts2.models.single_session.stat_baseline.base import StatBaseline
from ts2.models.single_session.stat_baseline.co_smoothing.readout_isi import RRRReadoutWithISI

BIN_SIZE = 0.02
CONTEXT_WINDOWS = [1.0, 5.0]


class FakeLinkDataset:
    """Duck-typed stand-in for the small slice of the dataset API ``link_datasets``
    (across every model below) actually reads: unit bookkeeping and the context window."""

    def __init__(
        self,
        context_window: float,
        bin_size: float = BIN_SIZE,
        num_units: int = 20,
        n_sessions: int = 1,
        target_window: float = 1.0,
    ):
        self.context_window = context_window
        self.BIN_SIZE = bin_size
        self.TARGET_WINDOW = target_window
        self._num_units = num_units
        self._session_ids = [f"sess{i}" for i in range(n_sessions)]
        self._unit_ids = [f"sess{i}/unit{u}" for i in range(n_sessions) for u in range(num_units)]

    def get_num_unit_per_recording(self):
        return dict.fromkeys(self._session_ids, self._num_units)

    def get_unit_ids(self):
        return list(self._unit_ids)


def _train_val(context_window: float, **kwargs) -> tuple[FakeLinkDataset, FakeLinkDataset]:
    return FakeLinkDataset(context_window, **kwargs), FakeLinkDataset(context_window, **kwargs)


@pytest.mark.parametrize("context_window", CONTEXT_WINDOWS)
def test_autoencoder_readout_scales_with_context_window(context_window):
    model = AutoencoderMLP()
    train, val = _train_val(context_window)
    model.link_datasets(train, val)
    T = round(context_window / BIN_SIZE)
    assert model.readout.out_features == T * len(train.get_unit_ids())


@pytest.mark.parametrize("context_window", CONTEXT_WINDOWS)
def test_ts2_ndt_position_embedding_scales_with_context_window(context_window):
    model = NDT(
        max_spikes=2,
        unit_emb_dim=0,
        encoder_num_heads=1,
        encoder_ffn_factor=1,
        encoder_num_layers=1,
        encoder_dropout=0.0,
        pre_encoder_dropout=0.0,
        post_encoder_dropout=0.0,
        encoder_activation="relu",
    )
    train, val = _train_val(context_window)
    model.link_datasets(train, val)
    assert model.num_bins == round(context_window / BIN_SIZE)
    assert model.position_emb.num_embeddings == model.num_bins


@pytest.mark.parametrize("context_window", CONTEXT_WINDOWS)
def test_ndt_superv_position_embedding_scales_with_context_window(context_window):
    model = NDTSuperv(
        bin_size=BIN_SIZE,
        max_spikes=2,
        spike_readin="none",
        hidden_dim=8,
        encoder_num_heads=1,
        encoder_ffn_factor=1,
        encoder_dropout=0.0,
        encoder_activation="relu",
        encoder_num_layers=1,
        pre_encoder_dropout=0.0,
        post_encoder_dropout=0.0,
    )
    train, val = _train_val(context_window)
    model.link_datasets(train, val)
    assert model.num_bins == round(context_window / BIN_SIZE)
    assert model.position_emb.num_embeddings == model.num_bins


def _stub_stat_baseline_fit(model, monkeypatch, num_units: int):
    """StatBaseline.link_datasets fits real windowed data after sizing self.T.
    Stub that part out so this stays a shape test, not an integration test."""
    monkeypatch.setattr(model, "_windows", lambda ds: np.zeros((1, model.T, num_units)))
    monkeypatch.setattr(model, "_mean_rate_val_score", lambda ds: 0.0)


@pytest.mark.parametrize("context_window", CONTEXT_WINDOWS)
def test_stat_baseline_T_scales_with_context_window(context_window, monkeypatch):
    model = StatBaseline()
    train, val = _train_val(context_window)
    train.task = val.task = "forecasting"
    train.FORECAST_RATIO = val.FORECAST_RATIO = 0.1
    _stub_stat_baseline_fit(model, monkeypatch, train._num_units)

    model.link_datasets(train, val)
    assert round(context_window / BIN_SIZE) == model.T


@pytest.mark.parametrize("context_window", CONTEXT_WINDOWS)
def test_readout_isi_inherits_stat_baseline_T_scaling(context_window, monkeypatch):
    """Co-smoothing readout: same StatBaseline.link_datasets, plus its own fit()."""
    model = RRRReadoutWithISI()
    train, val = _train_val(context_window)
    train.task = val.task = "co_smoothing"
    train.FORECAST_RATIO = val.FORECAST_RATIO = 0.1
    _stub_stat_baseline_fit(model, monkeypatch, train._num_units)
    monkeypatch.setattr(model, "fit", lambda *a, **kw: None)

    model.link_datasets(train, val)
    assert round(context_window / BIN_SIZE) == model.T
