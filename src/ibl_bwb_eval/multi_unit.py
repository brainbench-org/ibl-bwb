"""The multi-unit readout: TS3's second scored submission variant.

A submission may report per-unit probabilities directly, or pooled across neighbouring
units on the same probe. The scorer only takes an argmax, so the pooling rule is defined
here or nowhere.
"""

import numpy as np

__api_ref__ = {
    "description": None,
    "sections": [{"title": None, "autosummary": ["multi_unit_prediction"]}],
}

# Pool over units on the same probe within this distance, nearest first.
MULTI_UNIT_RADIUS_UM = 60.0
MULTI_UNIT_MAX_NEIGHBORS = 5


def multi_unit_prediction(
    pred_proba: np.ndarray,
    depths: np.ndarray,
    probe_ids: np.ndarray,
) -> np.ndarray:
    """Multi-unit averaging of prediction probabilities.

    Method introduced in NEMO. Each unit's probabilities are replaced by the mean over
    its :data:`MULTI_UNIT_MAX_NEIGHBORS` nearest units on the same probe within
    :data:`MULTI_UNIT_RADIUS_UM`, itself included.

    Args:
        pred_proba: prediction probabilities. Shape (units, n_classes)
        depths: depths on the probe for each unit. Shape (units,)
        probe_ids: string id of the probe for each unit. Shape (units,)

    Returns:
        averaged_proba: averaged probabilities. Shape (units, n_classes).
        Callers should apply argmax to obtain predicted labels.
    """
    averaged_proba = np.zeros_like(pred_proba)
    for i in range(len(pred_proba)):
        same_probe = probe_ids == probe_ids[i]
        depth_diffs = np.abs(depths - depths[i])
        within_range = depth_diffs <= MULTI_UNIT_RADIUS_UM
        eligible = np.where(same_probe & within_range)[0]
        sorted_idx = eligible[np.argsort(depth_diffs[eligible])][:MULTI_UNIT_MAX_NEIGHBORS]
        averaged_proba[i] = pred_proba[sorted_idx].mean(axis=0)

    return averaged_proba
