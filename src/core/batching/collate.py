from typing import Any

from torch_brain.batching import collate


def supervised_collate(batch: list[tuple[Any, dict]]):
    """Collate a batch of data consisting of inputs and targets for supervised learning.

    Args:
        batch: A list of tuples (X, y), each containing an input and a target.

    Returns:
        A tuple (X, y) containing the collated inputs and targets.
    """
    return (
        collate([sample[0] for sample in batch]),
        collate([sample[1] for sample in batch]),
    )
