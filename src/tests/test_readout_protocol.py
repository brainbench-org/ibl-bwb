"""`configure_readout` is typed by a protocol, so no suite owns the model interface.

The hook reads only `id`, `dim`, `target_resolution` and `num_timesteps`, a small suite-neutral
slice of `TS1ReadoutSpec`, so `ReadoutSpec` names that intersection instead.
`core` is then free of every suite's spec type, which is the property that regressed once
already: splitting the shared spec per suite left the universal base importing TS1's.
"""

import ast
from pathlib import Path

import pytest

from core.model import BaseModel
from ibl_bwb_eval.tasks import (
    SUITE_TASKS,
    TargetResolution,
    get_ts1_readout_spec,
    get_ts1_supported_tasks,
    get_ts3_readout_spec,
)

CORE = Path(__file__).resolve().parents[1] / "core"
SUITE_SPECS = {"TS1ReadoutSpec", "TS2ReadoutSpec", "TS3ReadoutSpec"}


def _specs():
    yield from ((f"ts1/{t}", get_ts1_readout_spec(t)) for t in get_ts1_supported_tasks())
    yield from ((f"ts3/{t}", get_ts3_readout_spec(t)) for t in SUITE_TASKS["ts3"])


@pytest.mark.parametrize("name,spec", list(_specs()), ids=lambda x: x if isinstance(x, str) else "")
def test_every_scored_task_satisfies_the_readout_protocol(name, spec):
    """A model typed against ReadoutSpec must work for any suite, so TS1 and TS3 qualify.

    TS2 is absent because it sizes no readout head: nothing calls ``configure_readout``
    for it, and ``TS2ReadoutSpec`` carries the held-out mask axis and the metrics instead
    of the slice this protocol names.
    """
    assert isinstance(spec.id, str), name
    assert isinstance(spec.dim, int) and spec.dim > 0, name
    assert isinstance(spec.target_resolution, TargetResolution), name
    assert isinstance(spec.num_timesteps, int) and spec.num_timesteps > 0, name


def test_ts3_targets_are_sequence_level():
    """A unit's spike train collapses to one target, the shape TS1's choice also has.

    Classification and a future CCF regression share this: they differ in what kind of
    value each target is, which is DataType, not in how many there are.
    """
    for task in SUITE_TASKS["ts3"]:
        assert get_ts3_readout_spec(task).target_resolution is TargetResolution.SEQUENCE, task


@pytest.mark.parametrize(
    "path", sorted(CORE.rglob("*.py")), ids=lambda p: str(p.relative_to(CORE.parent))
)
def test_core_names_no_suite_spec(path):
    """core is shared by all three suites, so it must not import any one suite's spec."""
    imported = {
        alias.asname or alias.name
        for node in ast.walk(ast.parse(path.read_text()))
        if isinstance(node, ast.ImportFrom)
        for alias in node.names
    }
    assert not (imported & SUITE_SPECS), f"{path.name} imports {sorted(imported & SUITE_SPECS)}"


def test_the_hook_is_annotated_with_the_protocol():
    ann = BaseModel.configure_readout.__annotations__["readout_spec"]
    name = getattr(ann, "__name__", str(ann))
    assert name == "ReadoutSpec", name
