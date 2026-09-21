"""ibl_bwb_eval must stay importable without the training stack.

External consumers install only the ``scoring`` extra, whose dependency set conflicts
with ``train`` (see the ``[tool.uv] conflicts`` block in pyproject.toml). These tests
fail the moment someone reaches from ibl_bwb_eval back into ``core``, a task suite, or a
training-only dependency, which is what would quietly break that install.
"""

import ast
import subprocess
import sys
from pathlib import Path

import pytest

BWB_EVAL = Path(__file__).resolve().parents[1] / "ibl_bwb_eval"
ROOT = Path(__file__).resolve().parents[2]
RELEASE_PYPROJECT = ROOT / "packaging" / "ibl-bwb-eval" / "pyproject.toml"

# What ibl_bwb_eval's own source may not import: the repo packages that ship with the
# training extra, and third-party names absent from the `scoring` extra. pandas is here
# because aggregation deliberately returns plain dicts rather than frames, so a real
# import would break the consumer install without failing anything else.
FORBIDDEN_ROOTS = {
    "core",
    "pandas",
    "pretrain",
    "ts1",
    "ts2",
    "ts3",
    "torch_brain",
    "hydra",
    "omegaconf",
    "ray",
    "optuna",
    "wandb",
}

# What importing it may not pull in, transitively included. pandas is not checked here:
# sklearn.metrics imports it opportunistically when it happens to be installed, which
# says nothing about ibl_bwb_eval.
RUNTIME_FORBIDDEN = FORBIDDEN_ROOTS - {"pandas"}


def _module_files():
    return sorted(BWB_EVAL.rglob("*.py"))


def _imported_roots(path: Path) -> set[str]:
    """Every top-level package name imported by a module, including deferred imports."""
    roots = set()
    for node in ast.walk(ast.parse(path.read_text())):
        if isinstance(node, ast.Import):
            roots.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            roots.add(node.module.split(".")[0])
    return roots


@pytest.mark.parametrize("path", _module_files(), ids=lambda p: p.name)
def test_no_import_of_training_packages(path):
    offenders = _imported_roots(path) & FORBIDDEN_ROOTS
    assert not offenders, f"{path.relative_to(BWB_EVAL.parent)} imports {sorted(offenders)}"


def test_importing_bwb_eval_does_not_load_training_packages():
    """A fresh interpreter importing the whole package must not touch the training stack."""
    code = (
        "import sys; import ibl_bwb_eval, ibl_bwb_eval.scoring.aggregation, "
        "ibl_bwb_eval.scoring.ts1, ibl_bwb_eval.scoring.ts2, ibl_bwb_eval.scoring.ts3; "
        f"loaded = {sorted(RUNTIME_FORBIDDEN)!r};"
        "print(','.join(m for m in loaded if m in sys.modules))"
    )
    out = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=True
    ).stdout.strip()
    assert out == "", f"importing ibl_bwb_eval loaded: {out}"


def _requirements(path: Path, key: str) -> list[str]:
    """Requirement strings from one ``key = [...]`` block. Hand-parsed because tomllib
    arrived in 3.11 and the supported floor is 3.10."""
    lines = path.read_text().splitlines()
    header = f"{key} = ["
    assert header in lines, f"{path} has no `{header}` block"
    out = []
    for line in lines[lines.index(header) + 1 :]:
        entry = line.strip()
        if entry == "]":
            return out
        if not entry.startswith("#"):
            out.append(entry.strip('",'))
    raise AssertionError(f"{path} has an unterminated `{key}` block")


def test_the_release_distribution_declares_the_scoring_extra():
    """The two lists live in separate files, so a pin bumped in one and not the other
    hands an external scorer a different environment than the one the suite is tested
    against. Compared verbatim, ordering included, so the two stay diffable."""
    assert _requirements(RELEASE_PYPROJECT, "dependencies") == _requirements(
        ROOT / "pyproject.toml", "scoring"
    )


def _name_and_version(req: str) -> tuple[str, str]:
    """``(name, version)`` for a pinned requirement, extras stripped."""
    name, _, version = req.partition("==")
    return name.partition("[")[0], version


def test_the_train_extra_matches_the_scoring_pins():
    """Training installs the scoring path and runs it, so every pin ``scoring`` declares
    has to be in ``train`` at the same version. Where one drifts, a developer and an
    external scorer score the same predictions with different code."""
    root = ROOT / "pyproject.toml"
    train = dict(_name_and_version(r) for r in _requirements(root, "train"))
    for name, version in (_name_and_version(r) for r in _requirements(root, "scoring")):
        assert name in train, f"`scoring` declares {name}, `train` does not"
        assert train[name] == version, (
            f"{name}: train pins {train[name] or '(unpinned)'}, scoring pins {version}"
        )


# Stdlib-only modules: reading the protocol or the task list must not cost a torch or
# numpy import, which is why ibl_bwb_eval/__init__.py loads PredictionsWriter lazily and why
# multi_unit_prediction sits outside tasks/.
LIGHT_MODULES = [
    "ibl_bwb_eval",
    "ibl_bwb_eval.protocol",
    "ibl_bwb_eval.tasks",
    "ibl_bwb_eval.tasks.types",
    "ibl_bwb_eval.tasks.ts1",
    "ibl_bwb_eval.tasks.ts2",
    "ibl_bwb_eval.tasks.ts3",
]


@pytest.mark.parametrize("module", LIGHT_MODULES)
@pytest.mark.parametrize("heavy", ["torch", "numpy"])
def test_the_light_modules_stay_light(module, heavy):
    """Measured as a delta, so an interpreter that preloads the module cannot fail this."""
    code = (
        f"import sys; before = {heavy!r} in sys.modules;"
        f" import {module};"
        f" print(before, {heavy!r} in sys.modules)"
    )
    before, after = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=True
    ).stdout.split()
    if before == "True":
        pytest.skip(f"this interpreter preloads {heavy}, so there is no delta to measure")
    assert after == "False", f"importing {module} pulled in {heavy}"
