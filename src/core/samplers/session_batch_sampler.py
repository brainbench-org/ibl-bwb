from collections.abc import Iterator

import torch
from torch_brain.datasets import DatasetIndex


class SessionBatchSampler(torch.utils.data.Sampler[list[DatasetIndex]]):
    """Group any :class:`DatasetIndex` sampler's output into per-session batches.

    All indices in a batch share the same session id. The inner sampler controls
    the order of indices within each session; ``shuffle_batches`` controls the
    order of batches across sessions.

    Args:
        sampler: Inner sampler whose ``__iter__`` yields :class:`DatasetIndex` objects.
        batch_size: Number of samples per batch.
        shuffle_batches: Whether to shuffle the final batch order. Defaults to False.
        generator: Generator used when shuffling batches. Defaults to None.
        drop_last: Whether to drop the last incomplete batch per session.
            Defaults to True.
    """

    def __init__(
        self,
        sampler: torch.utils.data.Sampler,
        batch_size: int,
        *,
        shuffle_batches: bool = False,
        generator: torch.Generator | None = None,
        drop_last: bool = True,
    ):
        if batch_size <= 0:
            raise ValueError("batch_size must be positive.")
        self.sampler = sampler
        self.batch_size = batch_size
        self.shuffle_batches = shuffle_batches
        self.generator = generator
        self.drop_last = drop_last
        self._batches_cache: list[list[DatasetIndex]] | None = None

    def _build_batches(self) -> list[list[DatasetIndex]]:
        indices_by_session: dict[str, list[DatasetIndex]] = {}
        for idx in self.sampler:
            session_id = idx.recording_id
            if session_id not in indices_by_session:
                indices_by_session[session_id] = []
            indices_by_session[session_id].append(idx)

        batches: list[list[DatasetIndex]] = []
        for indices in indices_by_session.values():
            max_len = len(indices)
            if self.drop_last:
                max_len = (max_len // self.batch_size) * self.batch_size
            for i in range(0, max_len, self.batch_size):
                batch = indices[i : i + self.batch_size]
                if len(batch) == self.batch_size or not self.drop_last:
                    batches.append(batch)

        return batches

    def _prepare_cache(self) -> None:
        if self._batches_cache is None:
            self._batches_cache = self._build_batches()

    def __len__(self) -> int:
        self._prepare_cache()
        return len(self._batches_cache)

    def __iter__(self) -> Iterator[list[DatasetIndex]]:
        self._prepare_cache()
        batches = self._batches_cache
        self._batches_cache = None  # reset so next epoch rebuilds

        if self.shuffle_batches and batches:
            for idx in torch.randperm(len(batches), generator=self.generator).tolist():
                yield batches[idx]
        else:
            yield from batches

    def set_epoch(self, epoch: int) -> None:
        """Pass the epoch to the underlying sampler if it supports it."""
        if hasattr(self.sampler, "set_epoch"):
            self.sampler.set_epoch(epoch)
