"""Shard-then-merge file cache for expensive per-recording features.

Workers write batches to their own shard files; a merge folds every shard into the
single cache file under an exclusive lock and replaces it atomically, so each record
is written twice rather than once per batch and a crash cannot leave a torn cache.
The payload format is the caller's: pass the serializer and the merge rule.
"""

import fcntl
import os
from collections.abc import Callable
from contextlib import contextmanager
from pathlib import Path
from typing import Any


class ShardedCache:
    """One shard-then-merge cache file, with the payload format left to the caller.

    Args:
        path: the cache file. Shards go in ``<stem>_shards/`` beside it, the lock in
            ``<stem>.lock``.
        suffix: file extension for the cache, its shards, and the temp file.
        dump: ``dump(payload, path)``, writes one file.
        load: ``load(path)``, reads one file back.
        merge: ``merge(chunks)``, folds the existing cache plus every kept shard into
            the payload to write. Chunks are ordered cache-first, then shard order.
        keep: optional per-shard predicate; a shard it rejects is dropped, not merged.
    """

    def __init__(
        self,
        path: Path | str,
        *,
        suffix: str,
        dump: Callable[[Any, Path], None],
        load: Callable[[Path], Any],
        merge: Callable[[list], Any],
        keep: Callable[[Any], bool] | None = None,
    ):
        self.path = Path(path)
        self.suffix = suffix
        self._dump = dump
        self._load = load
        self._merge = merge
        self._keep = keep or (lambda _chunk: True)

    @property
    def shard_dir(self) -> Path:
        return self.path.parent / f"{self.path.stem}_shards"

    @contextmanager
    def lock(self, exclusive: bool = True):
        """Serialize cache access, so concurrent ranks or jobs cannot see a half-written file."""
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.path.with_suffix(".lock"), "w") as f:
            fcntl.flock(f, fcntl.LOCK_EX if exclusive else fcntl.LOCK_SH)
            try:
                yield
            finally:
                fcntl.flock(f, fcntl.LOCK_UN)

    def exists(self) -> bool:
        return self.path.exists()

    def read(self):
        """Read the whole cache under a shared lock."""
        with self.lock(exclusive=False):
            return self._load(self.path)

    def write_shard(self, index: int, payload) -> None:
        """Write one batch to its own file, folded into the cache by :meth:`merge_shards`."""
        self.shard_dir.mkdir(parents=True, exist_ok=True)
        self._dump(payload, self.shard_dir / f"shard_{index:05d}{self.suffix}")

    def merge_shards(self) -> None:
        """Fold any shards into the cache and drop them. Caller holds the exclusive lock."""
        shards = (
            sorted(self.shard_dir.glob(f"shard_*{self.suffix}")) if self.shard_dir.is_dir() else []
        )
        if not shards:
            return

        chunks = [self._load(self.path)] if self.path.exists() else []
        chunks += [c for c in (self._load(shard) for shard in shards) if self._keep(c)]

        if chunks:
            tmp_path = self.path.with_suffix(f".tmp{self.suffix}")
            self._dump(self._merge(chunks), tmp_path)
            os.replace(tmp_path, self.path)  # atomic, so a crash cannot leave a torn cache

        for shard in shards:
            shard.unlink()
        self.shard_dir.rmdir()
