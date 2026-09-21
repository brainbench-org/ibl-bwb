from pathlib import Path
from typing import Literal, TypeAlias

from ibl_bwb_eval.protocol import EVAL_RECORDING_IDS

RecordingIdList: TypeAlias = Literal["pretrain", "eval"]

_HERE = Path(__file__).parent

# The eval list lives with the evaluation protocol in ibl_bwb_eval, so the scorer and the
# training code cannot disagree about which sessions are scored.
_LISTS: dict[RecordingIdList, Path] = {
    "eval": EVAL_RECORDING_IDS,
    "pretrain": _HERE / "pretrain_recording_ids.txt",
}


def read_recording_ids(name: RecordingIdList) -> list[str]:
    """Read one vendored list, in file order."""
    return [line.strip() for line in _LISTS[name].read_text().splitlines() if line.strip()]
