"""The unit and recording bookkeeping a build carries into every task suite."""

from .recording_ids import RecordingIdList, read_recording_ids
from .unit_filtering import (
    BuildUnitFiltering,
    UnitFiltering,
    UnitFilteringInfo,
    enforce_unit_filtering,
    read_build_unit_filtering,
)

__all__ = [
    "BuildUnitFiltering",
    "RecordingIdList",
    "UnitFiltering",
    "UnitFilteringInfo",
    "enforce_unit_filtering",
    "read_build_unit_filtering",
    "read_recording_ids",
]

__api_ref__ = {
    "description": None,
    "sections": [
        {
            "title": "Unit filtering",
            "autosummary": [
                "UnitFilteringInfo",
                "BuildUnitFiltering",
                "read_build_unit_filtering",
                "enforce_unit_filtering",
            ],
        },
        {
            "title": "Recording ids",
            "autosummary": ["read_recording_ids"],
        },
    ],
}
