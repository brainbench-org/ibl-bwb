"""MtM's unit-embedding extractor.

The embedding is the unit's pair of per-session stitcher columns, ``[in.T, out]``
concatenated. One detail matters here: ``in_stitcher`` projects to
``shared_proj_input_dim``, not to ``hidden_dim`` like
``out_stitcher`` does, so the two halves of a unit's embedding can have different widths.
Concatenation is still well-defined; only the equal-width assumption is not.
"""

import numpy as np
import torch

from core.dataset import WholeSessionSpikeDataset
from core.utils.logger import get_cli_logger
from ts3.models.transductive.base import TransductiveExtractor

logger = get_cli_logger()


class MtMExtractor(TransductiveExtractor):
    def read(
        self, state_dict: dict, dataset: WholeSessionSpikeDataset, uids: set[str]
    ) -> tuple[torch.Tensor, np.ndarray]:
        in_keys = [k for k in state_dict if k.startswith("in_stitcher") and k.endswith("weight")]
        out_keys = [k for k in state_dict if k.startswith("out_stitcher") and k.endswith("weight")]
        assert {k.split(".")[1] for k in in_keys} == {k.split(".")[1] for k in out_keys}, (
            "In and out stitcher keys must have the same session ids"
        )

        embs, out_uids = [], []
        for rec_id in sorted({k.split(".")[1] for k in in_keys}):
            if rec_id not in dataset.recording_ids:
                logger.warning(f"Recording {rec_id} not found in dataset")
                continue
            in_weight = state_dict[f"in_stitcher.{rec_id}.weight"].T  # (v, D_in)
            out_weight = state_dict[f"out_stitcher.{rec_id}.weight"]  # (v, D_out)
            assert in_weight.shape[0] == out_weight.shape[0], (
                f"In and out stitcher weights must cover the same units, got "
                f"{in_weight.shape[0]} and {out_weight.shape[0]}"
            )
            embs.append(torch.cat([in_weight, out_weight], dim=1))  # (v, D_in + D_out)

            rec_uids = dataset.get_recording(rec_id).units.id.astype(str)  # (v,)
            assert len(rec_uids) == in_weight.shape[0], (
                f"Number of units in recording {rec_id} ({len(rec_uids)}) does not match "
                f"the number of units in the stitcher weights ({in_weight.shape[0]})"
            )
            out_uids.extend(rec_uids)

        return torch.cat(embs, dim=0), np.array(out_uids)
