"""The ids a submission is aligned by, and how they reach a ``.safetensors`` file.

Ids travel as a uint8 matrix, one row per id, because safetensors holds tensors and not
strings. The row width is the tensor's own second dimension, so a file describes itself and
ids of any length round-trip: what a suite means by an entity (a unit today, a channel
later) never reaches this module. Shorter ids are padded with NUL, which ASCII ids cannot
contain, so a fixed-width batch pads to nothing and the padding is unambiguous.
"""

import numpy as np
import torch

__api_ref__ = {
    "description": None,
    "sections": [
        {"title": "Submission format", "autosummary": ["encode_entity_ids", "decode_entity_ids"]}
    ],
}

_PAD = b"\x00"


def encode_entity_ids(ids: np.ndarray) -> torch.Tensor:
    """Pack string ids into an (n, width) uint8 tensor, width being the longest id."""
    rows = [s.encode("ascii") for s in np.asarray(ids).astype(str)]
    if not rows:
        raise ValueError("no ids to encode")
    if not all(rows):
        raise ValueError("ids must not be empty")

    out = np.zeros((len(rows), max(map(len, rows))), dtype=np.uint8)
    for i, row in enumerate(rows):
        out[i, : len(row)] = np.frombuffer(row, dtype=np.uint8)
    return torch.from_numpy(out)


def decode_entity_ids(tensor: torch.Tensor) -> np.ndarray:
    """Unpack what :func:`encode_entity_ids` wrote, dropping the padding."""
    return np.array([row.tobytes().rstrip(_PAD).decode("ascii") for row in tensor.numpy()])
