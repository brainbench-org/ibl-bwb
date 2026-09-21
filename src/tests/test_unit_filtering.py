"""Tests for the unit-filter build label and the per-suite contract it gates.

No labeled build exists in the tree, so the two ends are tested against the source: the
pipeline's labeling functions are exec'd out of ``pipeline.py`` (it cannot be imported here,
it runs standalone with its own pinned deps), and each consumer's contract declaration is
read out of its module.
"""

import ast
from pathlib import Path
from types import SimpleNamespace
from typing import Literal, get_args

import numpy as np
import pytest

from core.data.unit_filtering import (
    DOCS,
    SELECTED_UNITS_FILTERS,
    BuildUnitFiltering,
    UnitFilteringInfo,
    enforce_unit_filtering,
    parse_filters,
    read_unit_filtering,
)
from core.utils.exceptions import DatasetContractError
from ibl_bwb_eval.protocol import EVAL_RECORDING_IDS

SRC = Path(__file__).parents[1]
PIPELINE = Path(__file__).parents[2] / "ibl_brain_wide_bench_2026" / "pipeline.py"
DATASET_DIR = Path("/data/ibl/ibl_brain_wide_bench_2026/eval")


def _recording(**brainset):
    return SimpleNamespace(brainset=SimpleNamespace(**brainset))


def _read(recording):
    return read_unit_filtering(recording, DATASET_DIR)


# ---------------------------------------------------------------- reading the label


def test_parse_filters_round_trips_the_stored_form():
    assert parse_filters(",".join(SELECTED_UNITS_FILTERS)) == SELECTED_UNITS_FILTERS
    assert parse_filters("") == ()
    assert parse_filters(None) == ()
    assert parse_filters(b"unit_qc,firing_rate") == ("firing_rate", "unit_qc")
    assert parse_filters(np.array(["unit_qc", "probe_qc"], dtype=object)) == (
        "probe_qc",
        "unit_qc",
    )


@pytest.mark.parametrize("label", ["all_units", "selected_units", "custom"])
def test_every_known_label_is_accepted(label):
    assert _read(_recording(unit_filtering=label)).label == label


def test_metadata_is_read_verbatim():
    info = _read(
        _recording(unit_filtering="selected_units", unit_filters=",".join(SELECTED_UNITS_FILTERS))
    )
    assert (info.label, info.filters) == ("selected_units", SELECTED_UNITS_FILTERS)


def test_build_without_the_label_is_refused_on_read():
    with pytest.raises(DatasetContractError) as excinfo:
        _read(_recording())

    message = str(excinfo.value)
    assert "records no unit_filtering" in message
    assert DOCS in message  # which variant to fetch depends on the suite


def test_unknown_label_is_refused_on_read():
    with pytest.raises(DatasetContractError) as excinfo:
        _read(_recording(unit_filtering="selected-units"))

    message = str(excinfo.value)
    assert "unit_filtering='selected-units'" in message
    assert "all_units, selected_units, custom" in message


def test_a_build_with_no_recordings_is_a_usage_error():
    with pytest.raises(ValueError, match="no recordings"):
        BuildUnitFiltering({})


# ---------------------------------------------------------------- contract


def _build(labels: dict[str, str], **kwargs) -> BuildUnitFiltering:
    return BuildUnitFiltering(
        {rid: UnitFilteringInfo(label=label, **kwargs) for rid, label in labels.items()}
    )


def _enforce(build, contract="TS2"):
    enforce_unit_filtering(
        build, required="selected_units", contract=contract, dataset_dir=DATASET_DIR
    )


def test_compliant_build_passes():
    _enforce(_build({"a": "selected_units"}, filters=SELECTED_UNITS_FILTERS))


def test_unfiltered_build_raises_a_diagnostic_error():
    with pytest.raises(DatasetContractError) as excinfo:
        _enforce(_build({"a": "all_units"}))

    message = str(excinfo.value)
    assert message.splitlines()[0].startswith("TS2 requires the selected_units dataset build")
    assert str(DATASET_DIR) in message
    assert "build:   all_units (filters applied: none)" in message
    assert "aws s3 sync" in message  # how to get the right build
    assert "--unit-filter all" in message  # how to rebuild it


def test_mixed_build_is_rejected_and_counted():
    build = _build(
        {"a": "selected_units", "b": "all_units", "c": "selected_units"},
        filters=SELECTED_UNITS_FILTERS,
    )
    assert build.label == "mixed"

    with pytest.raises(DatasetContractError) as excinfo:
        _enforce(build, contract="TS3 eval")

    message = str(excinfo.value)
    assert "is labeled mixed" in message
    assert "1 recordings are all_units: b" in message
    assert "2 recordings are selected_units" in message


# ---------------------------------------------------------------- contract declarations


def _declared_contracts(module: str) -> list[str]:
    """The ``require_unit_filtering`` values a module passes, as source text."""
    tree = ast.parse((SRC / module).read_text())
    return [
        ast.unparse(node.value)
        for node in ast.walk(tree)
        if isinstance(node, ast.keyword) and node.arg == "require_unit_filtering"
    ]


# Only modules that build a dataset themselves. A module that goes through
# WholeSessionSpikeDataset inherits the declaration that class makes for every caller,
# so it has nothing of its own to state here.
@pytest.mark.parametrize(
    "module",
    [
        "ts2/ts2_dataset.py",
        "core/dataset.py",
        "ts3/protocol.py",
    ],
)
def test_eval_builders_declare_the_contract(module):
    declared = _declared_contracts(module)
    assert declared, f"{module} builds an eval dataset without declaring a contract"
    assert all("selected_units" in value for value in declared), declared


def test_ts1_declares_no_contract():
    """TS1 is defined on the full population but accepts any build, by design."""
    assert _declared_contracts("ts1/ts1_dataset.py") == []


# ---------------------------------------------------------------- pipeline drift


def _pipeline_labeling() -> dict:
    """Exec the pipeline's own labeling helpers, without importing its heavy dependencies."""
    tree = ast.parse(PIPELINE.read_text())
    constants = {
        "UnitFilterType",
        "ALL_UNIT_FILTERS",
        "PROBE_QC_LABEL",
        "MIN_FIRING_RATE",
        "UNIT_QC_LABEL",
    }
    functions = {"resolve_unit_filters", "unit_filtering_label", "unit_filter_thresholds"}
    namespace = {"Literal": Literal, "get_args": get_args}

    for node in tree.body:
        wanted = (
            isinstance(node, ast.Assign)
            and any(getattr(t, "id", None) in constants for t in node.targets)
        ) or (isinstance(node, ast.FunctionDef) and node.name in functions)
        if wanted:
            exec(compile(ast.Module([node], []), str(PIPELINE), "exec"), namespace)

    missing = (constants | functions) - namespace.keys()
    assert not missing, f"{missing} no longer defined in {PIPELINE.name}"
    return namespace


def _pipeline_derived_version() -> str:
    """The derived_version literal the pipeline stamps on every recording."""
    for node in ast.walk(ast.parse(PIPELINE.read_text())):
        if isinstance(node, ast.keyword) and node.arg == "derived_version":
            return node.value.value
    raise AssertionError(f"derived_version not found in {PIPELINE.name}")


def test_pipeline_selected_units_matches_the_contract_message():
    """The one place the two copies of the filter set have to agree."""
    pipeline = _pipeline_labeling()
    assert pipeline["resolve_unit_filters"](["all"], False) == SELECTED_UNITS_FILTERS


@pytest.mark.parametrize(
    "requested,is_pretrain,expected",
    [
        ([], False, "all_units"),
        (["all"], False, "selected_units"),
        (["all", "firing_rate"], False, "selected_units"),  # a redundant name changes nothing
        (["firing_rate"], False, "custom"),
        (["probe_qc", "unit_qc"], False, "custom"),
        (["all"], True, "selected_units"),  # probe_qc dropped, the pair is canonical there
        (["probe_qc"], True, "all_units"),  # nothing left to apply
    ],
)
def test_pipeline_labels_a_build_by_what_it_applied(requested, is_pretrain, expected):
    pipeline = _pipeline_labeling()
    filters = pipeline["resolve_unit_filters"](requested, is_pretrain)
    assert pipeline["unit_filtering_label"](filters, is_pretrain) == expected


def test_pipeline_records_the_thresholds_it_applied():
    pipeline = _pipeline_labeling()
    assert pipeline["unit_filter_thresholds"](()) == ""
    assert pipeline["unit_filter_thresholds"](("firing_rate",)) == "firing_rate > 1.0"
    thresholds = pipeline["unit_filter_thresholds"](SELECTED_UNITS_FILTERS)
    assert thresholds == "firing_rate > 1.0, qc_neural == PASS, label == 1.0"


def test_pipeline_exposes_the_all_shorthand():
    source = PIPELINE.read_text()
    assert 'UNIT_FILTER_CHOICES = ("all", *ALL_UNIT_FILTERS)' in source
    assert "choices=UNIT_FILTER_CHOICES" in source


# ---------------------------------------------------------------- write / read round trip


@pytest.mark.parametrize(
    "requested,is_pretrain,expected",
    [
        (["all"], False, "selected_units"),
        (["all"], True, "selected_units"),
        ([], False, "all_units"),
    ],
)
def test_metadata_round_trips_from_the_pipeline_write(tmp_path, requested, is_pretrain, expected):
    """Write the brainset exactly as the pipeline does, then read it back."""
    import h5py
    from torch_brain.data import BrainsetDescription, Data, Interval, serialize_fn_map

    pipeline = _pipeline_labeling()
    filters = pipeline["resolve_unit_filters"](requested, is_pretrain)
    description = BrainsetDescription(
        id="ibl_brain_wide_bench_2026",
        origin_version="0.0.1",
        derived_version=_pipeline_derived_version(),
        source="one-api",
        unit_filtering=pipeline["unit_filtering_label"](filters, is_pretrain),
        unit_filters=",".join(filters),
        unit_filter_thresholds=pipeline["unit_filter_thresholds"](filters),
        description="...",
    )
    path = tmp_path / "recording.h5"
    with h5py.File(path, "w") as f:
        Data(brainset=description, domain=Interval(np.array([0.0]), np.array([1.0]))).to_hdf5(
            f, serialize_fn_map=serialize_fn_map
        )

    with h5py.File(path, "r") as f:
        info = read_unit_filtering(Data.from_hdf5(f), tmp_path)

    assert (info.label, info.filters) == (expected, filters)


# ---------------------------------------------------------------- local build, when present

PROCESSED = Path(__file__).parents[2] / "processed"
EVAL_IDS = EVAL_RECORDING_IDS


def _local_eval_recording() -> str | None:
    eval_dir = PROCESSED / "ibl_brain_wide_bench_2026" / "eval"
    if not eval_dir.is_dir():
        return None
    known = set(EVAL_IDS.read_text().split())
    local = sorted(p.stem for p in eval_dir.glob("*.h5") if p.stem in known)
    return local[0] if local else None


requires_local_build = pytest.mark.skipif(
    _local_eval_recording() is None, reason="no local processed build to read"
)


def _build_base():
    from core.dataset import IBLBrainWideBench2026

    dataset = IBLBrainWideBench2026(
        root=str(PROCESSED), regime="eval", split="test", recording_ids=[_local_eval_recording()]
    )
    return dataset.unit_filtering.label


def _build_ts2():
    from ts2.ts2_dataset import IBLBrainWideBenchTS2

    dataset = IBLBrainWideBenchTS2(
        root=str(PROCESSED),
        split="test",
        recording_id=_local_eval_recording(),
        task="co_smoothing",
    )
    return dataset.unit_filtering.label


@requires_local_build
@pytest.mark.parametrize("read", [_build_base, _build_ts2])
def test_a_real_build_is_labeled_or_refused_at_construction(read):
    """Never a verdict deferred to a DataLoader worker: it lands here or not at all."""
    try:
        assert read() in ("all_units", "selected_units", "custom")
    except DatasetContractError as error:
        assert DOCS in str(error)
