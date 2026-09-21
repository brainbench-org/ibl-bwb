"""Tests for the shard-then-merge cache and the ray fan-out behind the feature caches.

Both were three near-copies before, in NEMO's waveform/ACG cache, LOLCAT's ISI cache,
and the ISI feature extractor. What they encode is not obvious from a call site: a
merge folds the existing cache in first so a resumed build keeps what it had, the
replace is atomic so a crash cannot leave a torn cache, a shard the ``keep`` predicate
rejects is dropped rather than merged, and the fan-out yields exactly one value per
item so a caller's progress bar and its shard counter stay in step.
"""

import numpy as np
import pytest

from core.utils.ray_map import imap_unordered
from core.utils.shard_cache import ShardedCache


def _read_npz(path):
    data = np.load(path, allow_pickle=True)
    return {k: data[k] for k in data.files}


def _cache(path, **kwargs):
    kwargs.setdefault(
        "merge", lambda chunks: {k: np.concatenate([c[k] for c in chunks]) for k in chunks[0]}
    )
    return ShardedCache(
        path,
        suffix=".npz",
        dump=lambda payload, p: np.savez(p, **payload),
        load=_read_npz,
        **kwargs,
    )


def _rows(*names):
    return {"ids": np.array(names)}


def test_shards_merge_into_the_cache_and_are_removed(tmp_path):
    cache = _cache(tmp_path / "c.npz")
    cache.write_shard(0, _rows("a", "b"))
    cache.write_shard(1, _rows("c"))
    cache.merge_shards()

    assert cache.read()["ids"].tolist() == ["a", "b", "c"]
    assert not cache.shard_dir.exists()


def test_a_second_round_folds_into_what_is_already_cached(tmp_path):
    """A resumed build must keep the recordings the first build wrote."""
    cache = _cache(tmp_path / "c.npz")
    cache.write_shard(0, _rows("a"))
    cache.merge_shards()
    cache.write_shard(0, _rows("b"))
    cache.merge_shards()

    assert cache.read()["ids"].tolist() == ["a", "b"]


def test_merging_nothing_leaves_the_cache_untouched(tmp_path):
    cache = _cache(tmp_path / "c.npz")
    cache.write_shard(0, _rows("a"))
    cache.merge_shards()
    before = cache.path.read_bytes()

    cache.merge_shards()

    assert cache.path.read_bytes() == before


def test_a_rejected_shard_is_dropped_not_merged(tmp_path):
    """LOLCAT invalidates its cache on a params change; the stale shards must not survive."""
    cache = _cache(tmp_path / "c.npz", keep=lambda chunk: chunk["ids"][0] != "stale")
    cache.write_shard(0, _rows("a"))
    cache.write_shard(1, _rows("stale"))
    cache.merge_shards()

    assert cache.read()["ids"].tolist() == ["a"]
    assert not cache.shard_dir.exists()


def test_the_cache_is_replaced_atomically(tmp_path):
    """No temp file survives a merge, and the cache is only ever swapped in whole."""
    cache = _cache(tmp_path / "c.npz")
    cache.write_shard(0, _rows("a"))
    cache.merge_shards()

    assert [p.name for p in tmp_path.iterdir() if p.suffix == ".npz"] == ["c.npz"]


def test_the_lock_lives_beside_the_cache(tmp_path):
    cache = _cache(tmp_path / "sub" / "c.npz")
    with cache.lock():
        assert (tmp_path / "sub" / "c.lock").exists()


class _FakeRay:
    """Deterministic stand-in: a ref is its value, and wait completes in submission order."""

    @staticmethod
    def wait(futures, num_returns=1):
        return futures[:num_returns], futures[num_returns:]

    @staticmethod
    def get(ref):
        return ref.value


class _Ref:
    def __init__(self, value):
        self.value = value


@pytest.fixture(autouse=True)
def _fake_ray(monkeypatch):
    monkeypatch.setattr("core.utils.ray_map.ray", _FakeRay)


@pytest.mark.parametrize("cap", [None, 1, 3, 100])
def test_every_item_yields_exactly_one_value(cap):
    """Callers size a progress bar by len(items) and flush shards off this count."""
    assert len(list(imap_unordered(lambda x: _Ref(x), range(10), max_in_flight=cap))) == 10


@pytest.mark.parametrize("cap", [None, 1, 3, 100])
def test_a_declined_item_yields_none_and_a_returned_none_passes_through(cap):
    submit = lambda x: None if x % 3 == 0 else _Ref(None if x % 4 == 0 else x)  # noqa: E731
    out = list(imap_unordered(submit, range(12), max_in_flight=cap))

    assert len(out) == 12
    assert sorted(x for x in out if x is not None) == [1, 2, 5, 7, 10, 11]
    assert out.count(None) == 6  # 4 declined (0, 3, 6, 9) + 2 that returned None (4, 8)


def test_no_more_than_max_in_flight_are_queued():
    live = []
    submitted = []

    def submit(x):
        submitted.append(x)
        live.append(x)
        return _Ref(x)

    for done in imap_unordered(submit, range(20), max_in_flight=4):
        live.remove(done)
        assert len(live) <= 4


def test_an_uncapped_run_submits_everything_before_the_first_result():
    submitted = []
    gen = imap_unordered(lambda x: _Ref(submitted.append(x) or x), range(6))

    next(gen)

    assert submitted == list(range(6))


def test_empty_and_all_declined_inputs():
    assert list(imap_unordered(lambda x: _Ref(x), [])) == []
    assert list(imap_unordered(lambda x: None, range(3), max_in_flight=2)) == [None, None, None]
