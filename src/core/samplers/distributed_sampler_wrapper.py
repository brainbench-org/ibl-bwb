import math

import torch
import torch.distributed as dist


class DistributedSamplerWrapper(torch.utils.data.Sampler):
    """Wrapper for distributing any sampler across multiple processes.

    This wrapper takes an existing sampler and distributes its indices across
    multiple replicas (processes) in a distributed training setup. It ensures
    each replica gets a unique subset of the data by splitting the indices
    evenly across replicas.

    Note: This wrapper supports samplers that may return a different number of
    samples each epoch, as it recomputes the indices on each iteration.

    Note: If the length of the sampler is not a multiple of num_replicas, some
    samples will be dropped to ensure equal distribution across replicas.

    Args:
        sampler: The base sampler to wrap
        num_replicas: Number of processes in distributed training
        rank: Rank of the current process
    """

    def __init__(
        self,
        sampler: torch.utils.data.Sampler,
        num_replicas: int | None = None,
        rank: int | None = None,
    ):
        self.sampler = sampler

        if num_replicas is None:
            num_replicas = dist.get_world_size() if dist.is_initialized() else 1
        if rank is None:
            rank = dist.get_rank() if dist.is_initialized() else 0

        self.num_replicas = num_replicas
        self.rank = rank
        self._indices_cache = None

    def _prepare_indices_cache(self):
        if self._indices_cache is not None:
            return

        indices = list(self.sampler)

        # Force len(indices) to be a multiple of num_replicas
        num_samples_per_rank = math.floor(len(indices) / self.num_replicas)
        total_size = num_samples_per_rank * self.num_replicas
        indices = indices[:total_size]

        self._indices_cache = indices[self.rank : total_size : self.num_replicas]

    def __len__(self):
        self._prepare_indices_cache()
        return len(self._indices_cache)

    def __iter__(self):
        self._prepare_indices_cache()
        yield from self._indices_cache
        self._indices_cache = None

    def set_epoch(self, epoch: int):
        """Pass the epoch to the underlying sampler if it supports it."""
        if hasattr(self.sampler, "set_epoch"):
            self.sampler.set_epoch(epoch)
