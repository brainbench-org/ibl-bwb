"""Finding the transductive regime's finetuning checkpoints by scanning for them.

What the scan must not do is choose. Two runs of one recording and seed are identical in
everything a config shows, so picking either would settle a submission's contents by
directory order.
"""

import pytest
import torch

from ts3.models.transductive.ckpt_index import CKPT_NAME, resolve_finetune_ckpts

POYO = "pretrain.models.poyo.POYO"
STITCH = "pretrain.models.ndt_stitch.NDTStitch"


def _ckpt(root, run: str, *, seed: int, rec: str, model: str = POYO, **cfg) -> None:
    path = root / run / CKPT_NAME
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "cfg": {"seed": seed, "recording_ids": [rec], "model": {"_target_": model}, **cfg},
            "model_state_dict": {"w": torch.zeros(1)},
        },
        path,
    )


def _pretrain_ckpt(root, model: str = POYO):
    """A pretrain run holds every recording, which is what marks it as not a finetune."""
    path = root / "pretrain" / CKPT_NAME
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "cfg": {"seed": 42, "recording_ids": ["a", "b"], "model": {"_target_": model}},
            "model_state_dict": {"w": torch.zeros(1)},
        },
        path,
    )
    return path


def test_a_seed_resolves_to_one_checkpoint_per_recording(tmp_path):
    pre = _pretrain_ckpt(tmp_path)
    _ckpt(tmp_path, "r1", seed=1, rec="a")
    _ckpt(tmp_path, "r2", seed=1, rec="b")
    _ckpt(tmp_path, "r3", seed=2, rec="a")
    _ckpt(tmp_path, "r4", seed=2, rec="b")

    entries = resolve_finetune_ckpts(seed=1, ckpt_dir=str(tmp_path), pretrain_ckpt=pre)

    assert sorted(e.recording_id for e in entries) == ["a", "b"]
    assert all(CKPT_NAME in e.path for e in entries)


def test_the_pretrain_checkpoint_is_not_mistaken_for_a_finetune(tmp_path):
    """It holds every recording, so it cannot be one session's run."""
    pre = _pretrain_ckpt(tmp_path)
    _ckpt(tmp_path, "r1", seed=1, rec="a")

    entries = resolve_finetune_ckpts(seed=1, ckpt_dir=str(tmp_path), pretrain_ckpt=pre)

    assert [e.recording_id for e in entries] == ["a"]


def test_another_model_in_the_same_directory_is_ignored(tmp_path):
    """One ckpt_dir may hold several models' sweeps; the pretrain checkpoint says which."""
    pre = _pretrain_ckpt(tmp_path, model=POYO)
    _ckpt(tmp_path, "r1", seed=1, rec="a", model=POYO)
    _ckpt(tmp_path, "r2", seed=1, rec="a", model=STITCH)

    entries = resolve_finetune_ckpts(seed=1, ckpt_dir=str(tmp_path), pretrain_ckpt=pre)

    assert [e.path for e in entries] == [str(tmp_path / "r1" / CKPT_NAME)]


def test_two_runs_of_one_recording_and_seed_are_refused(tmp_path):
    """A rerun or an lr sweep left in place would otherwise be decided by sort order."""
    pre = _pretrain_ckpt(tmp_path)
    _ckpt(tmp_path, "first", seed=1, rec="a")
    _ckpt(tmp_path, "second", seed=1, rec="a")

    with pytest.raises(ValueError, match="more than one checkpoint"):
        resolve_finetune_ckpts(seed=1, ckpt_dir=str(tmp_path), pretrain_ckpt=pre)


def test_filters_pick_a_sweep_apart_without_moving_files(tmp_path):
    pre = _pretrain_ckpt(tmp_path)
    _ckpt(tmp_path, "keep", seed=1, rec="a", task="choice")
    _ckpt(tmp_path, "drop", seed=1, rec="a", task="wheel")

    entries = resolve_finetune_ckpts(
        seed=1, ckpt_dir=str(tmp_path), pretrain_ckpt=pre, filters={"task": "choice"}
    )

    assert [e.path for e in entries] == [str(tmp_path / "keep" / CKPT_NAME)]


def test_seeds_covering_different_recordings_are_refused(tmp_path):
    """Otherwise one seed scores fewer units than another and the spread is not comparable."""
    pre = _pretrain_ckpt(tmp_path)
    _ckpt(tmp_path, "r1", seed=1, rec="a")
    _ckpt(tmp_path, "r2", seed=1, rec="b")
    _ckpt(tmp_path, "r3", seed=2, rec="a")

    with pytest.raises(ValueError, match="cover different recordings"):
        resolve_finetune_ckpts(seed=1, ckpt_dir=str(tmp_path), pretrain_ckpt=pre)


def test_a_seed_with_no_runs_says_which_seeds_there_are(tmp_path):
    pre = _pretrain_ckpt(tmp_path)
    _ckpt(tmp_path, "r1", seed=1, rec="a")

    with pytest.raises(ValueError, match=r"No finetuning runs for seed 9; found \[1\]"):
        resolve_finetune_ckpts(seed=9, ckpt_dir=str(tmp_path), pretrain_ckpt=pre)


def test_a_ckpt_dir_that_is_not_there_says_so(tmp_path):
    """Rather than reporting an empty sweep, which sends you looking at the runs."""
    with pytest.raises(FileNotFoundError, match="is not a directory"):
        resolve_finetune_ckpts(seed=1, ckpt_dir=str(tmp_path / "typo"))


def test_a_csv_replaces_the_scan(tmp_path):
    """The escape hatch, for checkpoints the scan cannot reach or tell apart."""
    _ckpt(tmp_path, "r1", seed=1, rec="a")
    csv = tmp_path / "runs.csv"
    csv.write_text("ID,recording_ids,seed\nr1,\"['a']\",1\n")

    entries = resolve_finetune_ckpts(seed=1, ckpt_dir=str(tmp_path), csv_path=str(csv))

    assert [(e.recording_id, e.path) for e in entries] == [("a", str(tmp_path / "r1" / CKPT_NAME))]


def test_a_csv_naming_a_checkpoint_that_is_not_there_is_refused(tmp_path):
    csv = tmp_path / "runs.csv"
    csv.write_text("ID,recording_ids,seed\nmissing,\"['a']\",1\n")

    with pytest.raises(FileNotFoundError, match="do not exist"):
        resolve_finetune_ckpts(seed=1, ckpt_dir=str(tmp_path), csv_path=str(csv))
