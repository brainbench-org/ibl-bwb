"""Tests for the shared multi-seed runner in ``core.eval_seeds``.

Two things this locks down: the summary statistics both seed runners report, and the
fact that each seed trains from its own config. The suites' ``train_seeds.py`` used to
hold private copies of the first and disagree about the second.
"""

import numpy as np
import pandas as pd
import pytest
from omegaconf import OmegaConf

from core.eval_seeds import run_seeds, summarize_seeds


def test_summarize_uses_the_sample_std_and_its_sem():
    df = pd.DataFrame({"r2": [0.1, 0.2, 0.3]})
    mean, std, sem, n = summarize_seeds(df)["r2"]
    assert n == 3
    assert mean == pytest.approx(0.2)
    assert std == pytest.approx(np.std([0.1, 0.2, 0.3], ddof=1))
    assert sem == pytest.approx(std / np.sqrt(3))


def test_a_single_seed_reports_zero_spread_rather_than_nan():
    """ddof=1 over one value is undefined; the table must not print nan."""
    mean, std, sem, n = summarize_seeds(pd.DataFrame({"r2": [0.5]}))["r2"]
    assert (mean, std, sem, n) == (0.5, 0.0, 0.0, 1)


def test_seed_column_is_not_reported_as_a_metric():
    assert set(summarize_seeds(pd.DataFrame({"seed": [43, 44], "r2": [0.1, 0.2]}))) == {"r2"}


def test_missing_values_shrink_n_for_that_metric_only():
    df = pd.DataFrame({"r2": [0.1, np.nan, 0.3], "bps": [1.0, 2.0, 3.0]})
    stats = summarize_seeds(df)
    assert stats["r2"][3] == 2
    assert stats["bps"][3] == 3


def test_a_metric_present_but_never_populated_is_dropped():
    assert summarize_seeds(pd.DataFrame({"r2": [np.nan, np.nan]})) == {}


@pytest.fixture
def no_hydra_context(monkeypatch):
    """run_seeds records the Hydra choices; unit tests run outside a Hydra context."""

    class _Choices:
        runtime = type("runtime", (), {"choices": {"trainer": "mlp"}})()

    monkeypatch.setattr(
        "core.eval_seeds.HydraConfig", type("H", (), {"get": staticmethod(_Choices)})
    )


def test_each_seed_trains_from_its_own_config(no_hydra_context, monkeypatch):
    """A trainer writing into cfg must not leak into the next seed."""
    seen = []

    def fake_train(cfg, rank, world_size):
        seen.append((cfg.seed, cfg.num_epochs))
        cfg.num_epochs = 999  # a trainer mutating its config
        return {"r2": 0.1 * cfg.seed}

    monkeypatch.setattr("core.eval_seeds.train", fake_train)
    cfg = OmegaConf.create({"num_epochs": 10, "seed": 0})

    run_seeds(cfg, seeds=[43, 44, 45])

    assert seen == [(43, 10), (44, 10), (45, 10)]
    assert cfg.num_epochs == 10  # the caller's config is untouched too


def test_seeds_default_to_the_protocol_then_the_config(no_hydra_context, monkeypatch):
    from ibl_bwb_eval.protocol import EVAL_SEEDS

    seen = []
    monkeypatch.setattr(
        "core.eval_seeds.train", lambda cfg, r, w: seen.append(cfg.seed) or {"r2": 1.0}
    )

    run_seeds(OmegaConf.create({}))
    assert seen == EVAL_SEEDS

    seen.clear()
    run_seeds(OmegaConf.create({"eval_seeds": [1, 2]}))
    assert seen == [1, 2]
