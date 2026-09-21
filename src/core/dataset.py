"""The dataset every task suite and pretraining run reads the benchmark through."""

import dataclasses
import logging
from collections.abc import Callable
from dataclasses import dataclass
from functools import cached_property
from pathlib import Path
from typing import Any, Literal, TypeAlias, get_args

import numpy as np
from torch_brain.data import Data, Interval
from torch_brain.datasets import Dataset, DatasetIndex, SpikingDatasetMixin

from core.data import (
    BuildUnitFiltering,
    UnitFiltering,
    enforce_unit_filtering,
    read_build_unit_filtering,
    read_recording_ids,
)
from core.transforms.unit_mask import apply_unit_mask_

Split: TypeAlias = Literal["train", "val", "test"]
BenchmarkRegime: TypeAlias = Literal["pretrain", "eval"]
BrainRegionAtlas: TypeAlias = Literal["allen", "beryl", "cosmos"]
QCNeural: TypeAlias = Literal["PASS", "WARNING", "FAIL"]


__api_ref__ = {
    "description": None,
    "sections": [
        {
            "title": None,
            "autosummary": [
                "IBLBrainWideBench2026",
                "VariableContextMixin",
                "WholeSessionSpikeDataset",
                "UnitQCPolicy",
            ],
        },
    ],
}


class IBLBrainWideBench2026(SpikingDatasetMixin, Dataset):
    """Dataset for the IBL BrainWideBench benchmark.

    Args:
        root: The root directory of the dataset.
        dirname: The name of the dataset (and the directory containing its data).
        recording_ids: The recording ids to include in the dataset. If None, all
                       recordings in the dataset related to the regime are included.
        transform: The transform(s) to apply to the data.
        split: Which split to sample, one of the class's ``SPLITS``. None (default) for whole
            sessions.
        regime: The regime of the dataset (pretrain, eval).
        require_unit_filtering: Build label this consumer requires. None accepts any build.
        contract: Name used in the contract error, defaults to the class name.
        context_length: Total sampled window size in seconds (context preceding the
                        target). Must be less than or equal to MAX_CONTEXT_LENGTH.
    """

    spiking_dataset_mixin_uniquify_unit_ids = True

    # Upper bound on context_length. Matches the 20s split gap (MIN_SPLIT_GAP_SEC in
    # ibl_brain_wide_bench_2026/pipeline.py).
    MAX_CONTEXT_LENGTH = 20.0  # seconds

    # Which f"{scope}_{split}_domain" the build carries for this class
    SPLITS: tuple[Split | None, ...] = (*get_args(Split), None)

    def __init__(
        self,
        root: str,
        split: Split | None,
        dirname: str = "ibl_brain_wide_bench_2026",
        recording_ids: str | list[str] | None = None,
        transform: Any | None = None,
        regime: BenchmarkRegime = "pretrain",
        require_unit_filtering: UnitFiltering | None = None,
        contract: str | None = None,
        context_length: float = 1.0,
    ):
        assert regime in get_args(BenchmarkRegime), f"Unsupported regime: {regime}"
        if split not in self.SPLITS:
            raise ValueError(
                f"{type(self).__name__} does not define split={split!r}, "
                f"its splits are {self.SPLITS}"
            )
        assert context_length <= self.MAX_CONTEXT_LENGTH, (
            f"context_length={context_length} must be less than or equal to MAX_CONTEXT_LENGTH={self.MAX_CONTEXT_LENGTH}"
        )
        self.context_window = context_length

        regime_recording_ids = read_recording_ids(regime)

        self.dataset_dir = Path(root) / dirname / regime

        if isinstance(recording_ids, str):
            recording_ids = [recording_ids]

        if recording_ids is None or len(recording_ids) == 0:
            recording_ids = regime_recording_ids

        invalid = set(recording_ids) - set(regime_recording_ids)
        if invalid:
            raise ValueError(f"{sorted(invalid)} are not in the {regime} recordings")

        if regime == "pretrain" and len(recording_ids) != len(regime_recording_ids):
            logging.warning(
                f"Using {len(recording_ids)} out of {len(regime_recording_ids)} {regime} recordings"
            )

        self._regime = regime

        super().__init__(
            dataset_dir=self.dataset_dir,
            recording_ids=recording_ids,
            transform=transform,
            namespace_attributes=[
                "session.id",
                "subject.id",
                "units.id",
                "probes.id",
            ],
        )

        self.split = split
        self.return_target = False

        self._contract = contract or type(self).__name__
        if require_unit_filtering is not None:
            enforce_unit_filtering(
                self.unit_filtering,
                required=require_unit_filtering,
                contract=self._contract,
                dataset_dir=self.dataset_dir,
            )

    @cached_property
    def unit_filtering(self) -> BuildUnitFiltering:
        """Which unit filters this build ran."""
        build = read_build_unit_filtering(self)
        logging.info(f"Unit filtering ({self.regime}): {build.describe()}")
        return build

    @cached_property
    def dataset_version(self) -> str:
        """The data-build pipeline's ``derived_version``."""
        versions = {
            str(getattr(self.get_recording(rid).brainset, "derived_version", "unknown"))
            for rid in self.recording_ids
        }
        # validate that all versions are the same
        assert len(versions) == 1, f"Versions are not the same: {versions}"
        return versions.pop()

    @property
    def regime(self) -> BenchmarkRegime:
        return self._regime

    def __getitem__(self, index: DatasetIndex):
        data = self.get_recording(index.recording_id, "")
        sample = data.slice(index.start, index.end)
        sample = self.dataset_transform(sample)

        if self.return_target:
            target = self._get_target(sample)

        if self.transform is not None:
            sample = self.transform(sample)

        if self.return_target:
            return sample, target

        return sample

    def dataset_transform(self, data: Data) -> Data:
        """Defines dataset-level transformations that are applied to all recordings in the Dataset.

        This method can be applied on an entire recording or a single slice.

        Args:
            data: The Data object to apply the transformations to.

        Returns:
            The Data object with the transformations applied.

        Example:
            >>> dataset = IBLBrainWideBench2026(...)
            >>> data = dataset.get_recording(...)
            >>> data = dataset.dataset_transform(data)
            >>> slice = data.slice(0.5, 1.5)
            >>> slice = dataset.dataset_transform(slice)
        """
        return data

    def _get_target(self, data: Data):
        raise NotImplementedError

    def get_sampling_intervals(self):
        """Not defined here, since what is samplable depends on the suite.

        Raises:
            NotImplementedError: Always. A subclass names its own intervals.
        """
        raise NotImplementedError(
            "get_sampling_intervals is too general for the base benchmark dataset, "
            "look at the task suite level or create a more specific."
        )

    def get_session_ids(self) -> list[str]:
        ans = [self.get_recording(rid).session.id for rid in self.recording_ids]
        return np.sort(np.unique(ans)).tolist()

    def get_subject_ids(self) -> list[str]:
        ans = [self.get_recording(rid).subject.id for rid in self.recording_ids]
        return np.sort(np.unique(ans)).tolist()

    def get_num_unit_per_recording(self):
        return {rid: len(self.get_recording(rid).units) for rid in self.recording_ids}

    def get_brain_regions(self, atlas="cosmos") -> list[str]:
        """Return a sorted list of all brain regions across all recordings in the dataset."""
        assert atlas in get_args(BrainRegionAtlas), f"{atlas} is not a valid atlas."

        ans = [
            self.get_recording(rid).get_nested_attribute(f"units.region_{atlas}")
            for rid in self.recording_ids
        ]
        return np.sort(np.unique(np.concatenate(ans))).tolist()


class VariableContextMixin:
    """Extends the sampled window backward for extra context.

    For a suite with a fixed, trailing scored window. Mix in ahead of IBLBrainWideBench2026 (whose context_length/context_window/
    MAX_CONTEXT_LENGTH this builds on) and set TARGET_WINDOW to the suite's own
    fixed scored-window length.

    get_sampling_intervals() is untouched by this mixin: the eval set (which
    trials exist, what's scored) never changes with context_length. Extra context
    is added per sample in __getitem__, clipped to 0 near a recording or split
    boundary, never by mutating the shared Interval a sampler iterates over.
    """

    TARGET_WINDOW: float  # set by the concrete suite dataset

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)  # sets self.context_window
        assert self.context_window >= self.TARGET_WINDOW, (
            f"context_length={self.context_window} must be at least "
            f"TARGET_WINDOW={self.TARGET_WINDOW}"
        )

    def __getitem__(self, index):
        context_margin = self.context_window - self.TARGET_WINDOW
        if context_margin > 0:
            index = dataclasses.replace(index, start=max(index.start - context_margin, 0.0))
        return super().__getitem__(index)


@dataclass(frozen=True)
class UnitQCPolicy:
    """Which sessions and units a whole-session dataset keeps.

    Distinct from ``UnitFiltering``, which is what the build already did: this is the cut
    the consumer makes on top of it. Every dataset still names one, in its own directory,
    so a model that deviates does it on purpose.

    Every field asks whether a unit's data is usable, never whether it has a usable
    target, which is the task's business. Every field defaults to cutting nothing, so a
    policy states only what it removes and ``UnitQCPolicy()`` is the raw population.
    """

    keep_qc_neural: tuple[QCNeural, ...] = get_args(QCNeural)
    keep_qc_neural_alignment: tuple[QCNeural, ...] = get_args(QCNeural)
    min_firing_rate: float | None = None
    min_label: float | None = None

    def keep_mask(self, units) -> np.ndarray:
        """The units this policy keeps, over one recording's unit table."""
        keep = np.isin(units.qc_neural.astype(str), self.keep_qc_neural)
        keep &= np.isin(units.qc_neural_alignment.astype(str), self.keep_qc_neural_alignment)
        if self.min_firing_rate is not None:
            keep &= units.firing_rate >= self.min_firing_rate
        if self.min_label is not None:
            keep &= units.label >= self.min_label

        return keep


class WholeSessionSpikeDataset(IBLBrainWideBench2026):
    r"""The benchmark sampled over whole sessions, after neural QC.

    Unit-level pretraining and TS3 both come through here, so the units a model trains
    on cannot drift from the units a suite scores.

    Args:
        root: The root directory of the dataset.
        regime: The regime of the dataset (pretrain, eval).
        dirname: The name of the dataset (and the directory containing its data).
        recording_ids: Recordings to keep, a subset of the regime's list. None keeps all.
        transform: The transform(s) to apply to the data.
        unit_qc: Which sessions and units to keep. Defaults to the raw population;
            every consumer declares its own beside the dataset that uses it.
        contract: Name used in the contract error, defaults to the class name.
    """

    # no temporal split: TS3 splits by subject, not by time
    SPLITS: tuple[Split | None, ...] = (None,)

    def __init__(
        self,
        root: str,
        regime: BenchmarkRegime,
        unit_qc: UnitQCPolicy | None = None,
        dirname: str = "ibl_brain_wide_bench_2026",
        recording_ids: str | list[str] | None = None,
        transform: Callable | None = None,
        contract: str | None = None,
        **kwargs,
    ):

        self.unit_qc = UnitQCPolicy() if unit_qc is None else unit_qc

        super().__init__(
            root=root,
            transform=transform,
            regime=regime,
            recording_ids=recording_ids,
            split=None,
            dirname=dirname,
            require_unit_filtering="selected_units" if regime == "eval" else None,
            contract=contract,
            **kwargs,
        )

        self._drop_empty_recordings(named=recording_ids is not None)

    def _drop_empty_recordings(self, named: bool) -> None:
        """Drop the sessions the policy empties: a sampler cannot draw a unit from them.

        The only thing deciding which sessions survive, so it covers every policy rather
        than just the FAIL rule. A session the caller named is refused instead of dropped,
        since honouring part of a request is worse than refusing it.
        """
        kept = [
            r
            for r in self.recording_ids
            if self.unit_qc.keep_mask(self.get_recording(r).units).any()
        ]
        if len(kept) == len(self.recording_ids):
            return

        empty = sorted(set(self.recording_ids) - set(kept))
        if named or not kept:
            raise ValueError(f"{self.unit_qc} leaves no unit in {empty}")

        logging.warning(f"{self.unit_qc} empties {len(empty)} sessions, dropping them")
        self._recording_ids = kept

    def get_sampling_intervals(self) -> dict[str, Interval]:
        """The whole spike domain of each recording, keyed by recording id.

        A whole-session model reads a unit's entire train, so nothing is trimmed here.
        """
        ans = {}
        for rid in self.recording_ids:
            recording = self.get_recording(rid)
            ans[rid] = recording.spikes.domain  # ty:ignore[unresolved-attribute]

        return ans

    def dataset_transform(self, data: Data) -> Data:
        # the build label is enforced at construction; this is the per-unit cut on top
        return apply_unit_mask_(data, self.unit_qc.keep_mask(data.units))
