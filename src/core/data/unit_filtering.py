from dataclasses import dataclass
from pathlib import Path
from typing import Literal, TypeAlias, get_args

import numpy as np

from core.utils.exceptions import DatasetContractError

UnitFiltering: TypeAlias = Literal["all_units", "selected_units", "custom"]
"""The label a build carries: no filter, the canonical set for its regime, or anything else."""

UnitFilter: TypeAlias = Literal["probe_qc", "firing_rate", "unit_qc"]
"""One unit quality filter the pipeline can apply."""

# what the pipeline applies for --unit-filter all on an eval session, copied here for the
# contract message. Only eval builds are ever required, so its pretrain set is not needed.
SELECTED_UNITS_FILTERS: tuple[UnitFilter, ...] = ("firing_rate", "probe_qc", "unit_qc")

S3_BUCKET = "s3://brain-wide-bench/brainsets"
DOCS = "docs/source/guides/dataset.rst"


def parse_filters(raw) -> tuple[UnitFilter, ...]:
    """Read back the comma-joined ``unit_filters`` attribute."""
    if raw is None:
        return ()
    if isinstance(raw, bytes):
        raw = raw.decode()
    if not isinstance(raw, str):  # tolerate a list-valued attribute
        raw = ",".join(str(x) for x in np.atleast_1d(raw))
    return tuple(sorted(f for f in (p.strip() for p in raw.split(",")) if f))


def format_filters(filters: tuple[UnitFilter, ...]) -> str:
    return ", ".join(filters) if filters else "none"


@dataclass(frozen=True)
class UnitFilteringInfo:
    """The filter provenance of a single recording."""

    label: UnitFiltering
    filters: tuple[UnitFilter, ...] = ()

    def describe(self) -> str:
        return f"{self.label} (filters applied: {format_filters(self.filters)})"


def read_unit_filtering(recording, dataset_dir: Path) -> UnitFilteringInfo:
    """Read the label off ``brainset``.

    A build with no label, or one we do not recognize, is refused here: no consumer gets to
    decide what a build with no provenance might have been.
    """
    brainset = getattr(recording, "brainset", None)
    label = getattr(brainset, "unit_filtering", None)
    if label is None:
        # which variant to fetch depends on the suite, so point at the docs, not at a bucket
        raise DatasetContractError(
            f"the build at {dataset_dir} records no unit_filtering, so it predates the current "
            f"pipeline and cannot be used. Re-download or rebuild it, see {DOCS}"
        )

    label = label.decode() if isinstance(label, bytes) else str(label)
    if label not in get_args(UnitFiltering):
        raise DatasetContractError(
            f"the build at {dataset_dir} records unit_filtering={label!r}, which is not one of "
            f"{', '.join(get_args(UnitFiltering))}. See {DOCS}"
        )

    return UnitFilteringInfo(
        label=label, filters=parse_filters(getattr(brainset, "unit_filters", None))
    )


class BuildUnitFiltering:
    """The filter provenance of a whole build, which may be mixed across recordings."""

    def __init__(self, per_recording: dict[str, UnitFilteringInfo]):
        if not per_recording:
            raise ValueError("a build with no recordings has no unit filtering to report")
        self.per_recording = per_recording

    @property
    def labels(self) -> tuple[str, ...]:
        return tuple(sorted({info.label for info in self.per_recording.values()}))

    @property
    def is_mixed(self) -> bool:
        return len(self.labels) > 1

    @property
    def label(self) -> str:
        return "mixed" if self.is_mixed else self.labels[0]

    def recordings_with(self, label: str) -> list[str]:
        return sorted(r for r, info in self.per_recording.items() if info.label == label)

    def describe(self) -> str:
        if not self.is_mixed:
            return next(iter(self.per_recording.values())).describe()
        counts = ", ".join(f"{len(self.recordings_with(x))} {x}" for x in self.labels)
        return f"mixed ({counts})"


def read_build_unit_filtering(dataset) -> BuildUnitFiltering:
    """Read the label of every recording the dataset covers."""
    return BuildUnitFiltering(
        {
            rid: read_unit_filtering(dataset.get_recording(rid), dataset.dataset_dir)
            for rid in dataset.recording_ids
        }
    )


def enforce_unit_filtering(
    build: BuildUnitFiltering,
    required: UnitFiltering,
    contract: str,
    dataset_dir: Path,
) -> None:
    """Raise unless the build satisfies the contract. There is no way to opt out."""
    if build.label != required:
        raise DatasetContractError(_contract_message(build, required, contract, dataset_dir))


def _contract_message(
    build: BuildUnitFiltering,
    required: UnitFiltering,
    contract: str,
    dataset_dir: Path,
) -> str:
    lines = [
        f"{contract} requires the {required} dataset build, "
        f"but the build at {dataset_dir} is labeled {build.label}.",
        f"  build:   {build.describe()}",
        f"  needed:  {required} (filters: {format_filters(SELECTED_UNITS_FILTERS)})",
    ]

    if build.is_mixed:
        for label in build.labels:
            offenders = build.recordings_with(label)
            shown = ", ".join(offenders[:3]) + (" ..." if len(offenders) > 3 else "")
            lines.append(f"    {len(offenders)} recordings are {label}: {shown}")

    lines += rebuild_hints(dataset_dir, required)
    return "\n".join(lines)


def rebuild_hints(dataset_dir: Path, variant: UnitFiltering = "selected_units") -> list[str]:
    """The re-download / rebuild / verify lines shared by every contract error."""
    dirname, regime = dataset_dir.parent.name, dataset_dir.name
    return [
        f"  fix:     aws s3 sync {S3_BUCKET}/{variant}/{dirname}/{regime}/ "
        f"{dataset_dir}/ --no-sign-request",
        "  or:      brainsets prepare ./ibl_brain_wide_bench_2026 --local "
        "--processed-dir <dir> --unit-filter all",
        f"  docs:    {DOCS}",
    ]
