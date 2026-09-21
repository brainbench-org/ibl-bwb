"""Which seed a TS3 submission is filed under, and where that number comes from.

The seed names the file, so it has to be whatever differs between two runs of one model:
the finetuning seed for the transductive extractors, whose pretrain checkpoint is shared,
and the pretraining seed for the inductive ones. An extractor reporting a constant would
collapse a sweep into one submission, and every file would still look well formed.
"""

import numpy as np
import pytest
import torch
from omegaconf import DictConfig, OmegaConf

from ts3.eval import _submission_seed
from ts3.models.inductive import ISIExtractor, NEMOExtractor, NuCLRExtractor
from ts3.models.transductive import (
    MtMExtractor,
    NDTStitchExtractor,
    POSSMExtractor,
    POYOPlusExtractor,
    TransductiveExtractor,
)

TRANSDUCTIVE = [POYOPlusExtractor, POSSMExtractor, NDTStitchExtractor, MtMExtractor]
INDUCTIVE = [NEMOExtractor, NuCLRExtractor]


@pytest.mark.parametrize("cls", TRANSDUCTIVE, ids=lambda c: c.__name__)
def test_two_finetuning_seeds_cannot_share_a_file(cls):
    """The pretrain checkpoint is shared, so the finetuning seed is the one that varies."""
    one = cls(pretrain_ckpt="run/best.pt", seed=1, ckpt_dir="ckpt")
    two = cls(pretrain_ckpt="run/best.pt", seed=2, ckpt_dir="ckpt")

    assert (one.run_seed, two.run_seed) == (1, 2)
    assert one.name != two.name  # nor an embeddings file


@pytest.mark.parametrize("cls", INDUCTIVE, ids=lambda c: c.__name__)
def test_an_inductive_extractor_reports_the_checkpoint_it_read(cls):
    """Read off the config the checkpoint carries, so no one has to be told it twice."""
    extractor = cls(ckpt="run/best.pt")
    extractor.cfg = DictConfig({"seed": 3})  # what setup() loads out of the checkpoint

    assert extractor.run_seed == 3


def test_a_training_free_extractor_claims_no_seed():
    """No run stands behind an ISI histogram, so the submitter names the submission."""
    assert ISIExtractor().run_seed is None


def test_only_the_checkpoint_can_confirm_which_seed_trained_it(tmp_path):
    """Two seeds of one recording are otherwise indistinguishable: same shapes, same uids."""
    read_calls = []

    class _Reader(TransductiveExtractor):
        def read(self, state_dict, dataset, uids):
            read_calls.append(state_dict)
            return torch.zeros(0), np.array([])

    extractor = _Reader(pretrain_ckpt="run/best.pt", seed=1, ckpt_dir="ckpt")
    weights = {"model_state_dict": {"w": torch.zeros(1)}}
    mine, theirs = tmp_path / "mine.pt", tmp_path / "theirs.pt"
    torch.save({"cfg": {"seed": 1}, **weights}, mine)
    torch.save({"cfg": {"seed": 2}, **weights}, theirs)

    extractor._read_ckpt(mine, None, set(), expect_seed=1)
    with pytest.raises(AssertionError, match="trained with seed 2, not 1"):
        extractor._read_ckpt(theirs, None, set(), expect_seed=1)

    assert len(read_calls) == 1  # the mismatch never reached the weights


@pytest.mark.parametrize(
    "cfg_seed,file_seed,expected",
    [(None, 7, 7), (7, 7, 7), (3, None, 3)],
    ids=["the file's own", "both, agreeing", "the config, for a file carrying none"],
)
def test_where_the_submission_seed_comes_from(cfg_seed, file_seed, expected):
    assert _submission_seed(OmegaConf.create({"seed": cfg_seed}), file_seed) == expected


def test_a_seed_that_disagrees_with_the_file_is_refused():
    """Otherwise one run's predictions are filed under another run's name."""
    with pytest.raises(ValueError, match="disagrees"):
        _submission_seed(OmegaConf.create({"seed": 2}), 7)


def test_a_file_that_records_nothing_needs_the_seed_passed():
    with pytest.raises(AssertionError, match="seed is required"):
        _submission_seed(OmegaConf.create({"seed": None}), None)
