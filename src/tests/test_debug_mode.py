"""Tests for ``cfg.debug``, the one-batch smoke run.

The cap reaches the ~18 trainers that each write their own ``train_epoch`` by wrapping
the loaders they share rather than the loops they do not, so what has to hold is that a
wrapped loader is indistinguishable from the real one apart from how far it iterates.
A loop reading ``len`` for a progress bar, or ``.dataset`` to link a model, sees no
difference.
"""

from pathlib import Path

import pytest
import torch
import yaml
from torch.utils.data import DataLoader, TensorDataset

from core.trainer import _OneBatchLoader

SRC = Path(__file__).parents[1]
SUITES = ["ts1", "ts2", "ts3", "pretrain"]


@pytest.fixture
def loader():
    return DataLoader(TensorDataset(torch.arange(20)), batch_size=2)  # 10 batches


def test_iteration_stops_after_one_batch(loader):
    assert len(list(_OneBatchLoader(loader))) == 1


def test_len_reports_one_batch(loader):
    assert len(_OneBatchLoader(loader)) == 1


def test_an_empty_split_stays_empty():
    empty = DataLoader(TensorDataset(torch.empty(0)), batch_size=2)

    assert len(_OneBatchLoader(empty)) == 0


def test_the_batch_is_the_loader_s_first(loader):
    ((batch,),) = _OneBatchLoader(loader)

    assert torch.equal(batch, next(iter(loader))[0])


def test_it_is_reiterable(loader):
    one_batch = _OneBatchLoader(loader)

    assert len(list(one_batch)) == len(list(one_batch)) == 1


def test_everything_else_passes_through(loader):
    one_batch = _OneBatchLoader(loader)

    assert one_batch.dataset is loader.dataset
    assert one_batch.batch_size == loader.batch_size


@pytest.mark.parametrize("suite", SUITES)
def test_debug_is_off_by_default(suite):
    """A suite that shipped it on would silently stop checkpointing every run."""
    cfg = yaml.safe_load((SRC / suite / "configs" / "train.yaml").read_text())

    assert cfg["debug"] is False
