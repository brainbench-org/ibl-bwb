from collections.abc import Callable
from copy import deepcopy

import numpy as np
from torch_brain.data import Interval
from torch_brain.datasets import DatasetIndex

from core.dataset import BenchmarkRegime, UnitQCPolicy, WholeSessionSpikeDataset

# What NuCLR pretrains on, overridable per run since nothing downstream is keyed to it.
NUCLR_UNIT_QC = UnitQCPolicy(keep_qc_neural=("PASS", "WARNING"))


class NuCLRDataset(WholeSessionSpikeDataset):
    """Modification of WholeSessionSpikeDataset to work with NuCLR's contrastive training

    Args:
        root: The root directory of the dataset.
        regime: The regime of the dataset (pretrain, eval).
        input_fn: The model input function
        dirname: The name of the dataset (and the directory containing its data).
        transform: The transform(s) to apply to the data.
        unit_qc: Which sessions and units to keep. NuCLR pretrains on what QC leaves.
    """

    def __init__(
        self,
        root: str,
        regime: BenchmarkRegime,
        input_fn: Callable,
        dirname: str = "ibl_brain_wide_bench_2026",
        transform: Callable | None = None,
        unit_qc: UnitQCPolicy = NUCLR_UNIT_QC,
        **kwargs,
    ):

        super().__init__(
            unit_qc=unit_qc,
            root=root,
            regime=regime,
            dirname=dirname,
            transform=transform,
            **kwargs,
        )

        self.input_fn = input_fn
        self._two_view = False

    def enable_two_views(self, max_distance: float | None, seed: int):
        """Prepare the dataset to return two views on every __getitem__

        Args:
            max_distance: Max time delta between the two views
            seed: integer seed for numpy RNG
        """
        # Two view mode needs to keep sampling_intervals around to able to generate
        # valid start/end times for the second view. So, we load them once and cache it
        self._sampling_intervals = self.get_sampling_intervals()
        self._two_view = True
        self.max_distance = max_distance or float("inf")
        self._rng = np.random.default_rng(seed)

    def get_sampling_intervals(self) -> dict[str, Interval]:
        if self._two_view:
            return self._sampling_intervals

        return super().get_sampling_intervals()

    def _get_second_view(self, v: DatasetIndex):
        if self.max_distance == 0.0:
            return v

        dur = v.end - v.start
        intrvl = self._sampling_intervals[v.recording_id]

        start = self._rng.uniform(
            max(v.start - self.max_distance, intrvl.start[0]),
            min(v.end + self.max_distance - dur, intrvl.end[-1] - dur),
        )
        end = start + dur

        assert start >= intrvl.start[0]  # ty:ignore[unresolved-attribute]
        assert end <= intrvl.end[-1]  # ty:ignore[unresolved-attribute]

        ans = deepcopy(v)
        ans.start = start
        ans.end = end
        return ans

    def __getitem__(self, index: DatasetIndex):
        X1 = self.input_fn(super().__getitem__(index))
        if not self._two_view:
            return X1

        if X1 is None:
            return None

        index2 = self._get_second_view(index)
        X2 = self.input_fn(super().__getitem__(index2))

        if X2 is None:
            return None

        return X1, X2
