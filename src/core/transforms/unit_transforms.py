import numpy as np
from torch_brain.data import Data

from core.transforms.unit_mask import apply_unit_mask_


class UnitDropout:
    """A simple unit dropout transform.

    It first samples the proportion of units to keep by uniformly sampling
    a float ``p`` in [``min_units``, 1.0]. Then it removes (1 - ``p``) units
    randomly.

    Args:
        min_units: Minimum proportion of units to keep
    """

    def __init__(self, min_units: float):
        assert min_units <= 1.0
        self.min_units = min_units

    def __call__(self, data) -> Data:
        num_units_before = len(data.units)
        min_units = int(num_units_before * self.min_units)

        if num_units_before <= min_units:
            return data

        num_units_after = np.random.randint(min_units, num_units_before + 1)
        sampled_unit_idx = np.random.permutation(num_units_before)[:num_units_after]
        sampled_unit_mask = np.zeros(num_units_before, dtype=bool)
        sampled_unit_mask[sampled_unit_idx] = True

        return apply_unit_mask_(data, sampled_unit_mask)


class FilterUnits:
    """Drops units that have failing probe QC or did not fire in this slice."""

    def __call__(self, data: Data) -> Data:
        _ = data.spikes.timestamps  # must read timestamps before unit_index (temporaldata bug)
        unit_idx = data.spikes.unit_index

        active_unit_idx = np.unique(unit_idx)
        mask = np.zeros(len(data.units), dtype=bool)
        mask[active_unit_idx] = True

        # Probe QC fail
        mask &= data.units.qc_neural.astype(str) != "FAIL"

        apply_unit_mask_(data, mask)

        assert len(data.units) == len(np.unique(data.spikes.unit_index))
        return data
