"""The entity-id byte matrix, and the width it must not assume.

Ids reach a submission as a uint8 matrix because safetensors holds tensors and not strings.
The width used to be a module constant, so ids of another length were silently reinterpreted
into a different number of scrambled ids rather than rejected: 73 ids of width 10 decoded as
10 ids of width 73, with no error. The width now comes from the data, and these tests pin it.
"""

import numpy as np
import pytest
import torch

from ibl_bwb_eval.entity_ids import decode_entity_ids, encode_entity_ids


def _uids(n: int) -> np.ndarray:
    """n ids in today's 73-char '{session}/{unit}' shape."""
    return np.array([f"{'a' * 36}/{i:036d}" for i in range(n)])


def test_todays_fixed_width_ids_are_unchanged():
    """The 73-char format predates the padding, so it must pad to nothing."""
    ids = _uids(5)
    tensor = encode_entity_ids(ids)
    assert tensor.shape == (5, 73)
    assert (tensor != 0).all(), "a fixed-width batch must carry no padding"
    assert (decode_entity_ids(tensor) == ids).all()


def test_variable_width_ids_round_trip():
    """What a channel id needs: an index of 1 to 3 digits, no zero-padding required."""
    ids = np.array(["probe00/0", "probe00/42", "probe00/383"])
    tensor = encode_entity_ids(ids)
    assert tensor.shape == (3, 11)
    assert (decode_entity_ids(tensor) == ids).all()


def test_width_is_derived_not_assumed():
    """The old failure: N * width divisible by 73 reinterpreted rather than raised."""
    ids = np.array([f"probe00/{i:02d}" for i in range(73)])
    assert len(ids) * len(ids[0]) % 73 == 0, "this batch is the one that used to reinterpret"
    assert (decode_entity_ids(encode_entity_ids(ids)) == ids).all()


def test_a_prefix_id_stays_distinct_from_the_id_it_prefixes():
    """Padding must not make a short id decode as its longer neighbour."""
    ids = np.array(["probe00/1", "probe00/10"])
    assert list(decode_entity_ids(encode_entity_ids(ids))) == ["probe00/1", "probe00/10"]


@pytest.mark.parametrize(
    ("ids", "match"),
    [
        (np.array([], dtype=str), "no ids"),
        (np.array(["ok", ""]), "must not be empty"),
    ],
)
def test_unencodable_ids_raise(ids, match):
    with pytest.raises(ValueError, match=match):
        encode_entity_ids(ids)


def test_non_ascii_ids_raise():
    with pytest.raises(UnicodeEncodeError):
        encode_entity_ids(np.array(["probe00/µ"]))


def test_decode_reads_the_width_off_the_tensor():
    """Decoding takes no constant, so a file written at any width still reads."""
    tensor = torch.tensor([[97, 98, 0], [99, 100, 101]], dtype=torch.uint8)
    assert list(decode_entity_ids(tensor)) == ["ab", "cde"]
