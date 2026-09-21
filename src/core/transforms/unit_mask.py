import numpy as np
from torch_brain.data import Data


def apply_unit_mask_(data: Data, mask: np.ndarray) -> Data:
    """Mask units and corresponding spikes given a boolean mask (in-place).

    Args:
        data: The data object to apply the unit mask to
        mask: The mask (boolean numpy array). True means that unit is preserved.

    Returns:
        The same data object as the input. This is an inplace operation
    """
    if mask.all():
        return data

    num_units_before = len(data.units)
    data.units = data.units.select_by_mask(mask)
    num_units_after = len(data.units)

    old_unit_idx = np.arange(num_units_before)[mask]
    unit_idx_remap = np.zeros(num_units_before, dtype=int)
    unit_idx_remap[old_unit_idx] = np.arange(num_units_after)

    spike_mask = np.isin(data.spikes.unit_index, old_unit_idx)
    data.spikes = data.spikes.select_by_mask(spike_mask)
    data.spikes.unit_index = unit_idx_remap[data.spikes.unit_index]

    return data
