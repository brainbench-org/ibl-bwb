"""Tests for the unit-embeddings file, the handoff between pretraining and the TS3 probes.

One writer (``ts3/extract.py``, whichever extractor it runs) and one reader (the probes)
agree on the key names and nothing else. Nothing validates the file at
either end, so a writer that drops a key or transposes uids against embeddings produces
a file that loads and scores, just against the wrong labels. The run seed is the one key a
reader may find missing, since a file from outside this repo carries none.
"""

import numpy as np
import pytest
import torch

from core.utils.unit_embeddings import load_unit_embeddings, save_unit_embeddings

UIDS = ["a", "b", "c"]


def _embs(n: int = 3, d: int = 4) -> torch.Tensor:
    return torch.arange(n * d, dtype=torch.float32).reshape(n, d)


def test_what_is_written_is_what_is_read(tmp_path):
    train, evl = _embs(), _embs() + 100
    save_unit_embeddings(tmp_path / "e.pt", train, UIDS, evl, ["x", "y", "z"])
    train_embs, train_uids, eval_embs, eval_uids, _ = load_unit_embeddings(tmp_path / "e.pt")

    assert np.array_equal(train_embs, train.numpy())
    assert np.array_equal(eval_embs, evl.numpy())
    assert train_uids.tolist() == UIDS
    assert eval_uids.tolist() == ["x", "y", "z"]


def test_the_run_seed_rides_along_and_is_none_when_unset(tmp_path):
    """It names the submission file, so a probe must not have to be told it twice."""
    save_unit_embeddings(tmp_path / "seeded.pt", _embs(), UIDS, _embs(), UIDS, run_seed=7)
    save_unit_embeddings(tmp_path / "plain.pt", _embs(), UIDS, _embs(), UIDS)

    assert load_unit_embeddings(tmp_path / "seeded.pt")[-1] == 7
    assert load_unit_embeddings(tmp_path / "plain.pt")[-1] is None


def test_a_file_without_the_key_reads_as_no_seed(tmp_path):
    """Every embeddings file written before the key existed still loads."""
    path = tmp_path / "old.pt"
    save_unit_embeddings(path, _embs(), UIDS, _embs(), UIDS, run_seed=7)
    old = torch.load(path, weights_only=False)
    del old["run_seed"]
    torch.save(old, path)

    assert load_unit_embeddings(path)[-1] is None


def test_a_directory_is_named_by_run_id(tmp_path):
    path = save_unit_embeddings(tmp_path / "embs", _embs(), UIDS, _embs(), UIDS, run_id="abc123")
    assert path == tmp_path / "embs" / "abc123.pt"
    assert path.is_file()


def test_a_directory_without_a_run_id_is_refused(tmp_path):
    with pytest.raises(ValueError, match="run_id"):
        save_unit_embeddings(tmp_path / "embs", _embs(), UIDS, _embs(), UIDS)


def test_a_graph_bound_tensor_is_saved_without_its_graph(tmp_path):
    """Trainers call this on live activations, so the file must not carry a graph back."""
    train = _embs().requires_grad_()
    path = save_unit_embeddings(tmp_path / "e.pt", train * 2, UIDS, _embs(), UIDS)
    assert not torch.load(path, weights_only=False)["train_embs"].requires_grad


def test_a_unit_whose_embedding_is_not_finite_is_dropped_with_its_uid(tmp_path):
    """Dropping the row but not the uid is the silent failure: every later row shifts."""
    train, evl = _embs(), _embs()
    train[1, 0] = float("nan")
    evl[2, 3] = float("inf")
    save_unit_embeddings(tmp_path / "e.pt", train, UIDS, evl, UIDS)
    train_embs, train_uids, eval_embs, eval_uids, _ = load_unit_embeddings(tmp_path / "e.pt")

    assert train_uids.tolist() == ["a", "c"]
    assert np.array_equal(train_embs, train[[0, 2]].numpy())
    assert eval_uids.tolist() == ["a", "b"]
    assert np.array_equal(eval_embs, evl[[0, 1]].numpy())
