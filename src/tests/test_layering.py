"""The import graph runs one way: ibl_bwb_eval, then core, then pretrain, then a suite.

Each layer has to be readable without the ones above it. That is what lets a participant
work inside a single suite directory, and what keeps eval-side contracts out of
``pretrain/``, where a model that never sees the scored units cannot accidentally be
tuned on them. Nothing checked this before and ``pretrain`` had already grown five
imports of ``ts3``, the last two being Nemo's and NuCLR's online region probe. There are
none now, and no allowance for one: a new violation is fixed rather than listed here.

The scan is AST based, so an import deferred inside a function counts the same as one at
module scope. Only Python is checked: a Hydra ``_target_`` naming a suite from a pretrain
config picks an implementation at run time rather than creating a source dependency, and
stays legal. ``ibl_bwb_eval``'s runtime isolation is checked in
test_ibl_bwb_eval_isolation.py, which also covers its third-party dependency set.
"""

import ast
from pathlib import Path

import pytest

SRC = Path(__file__).resolve().parents[1]

# Package -> the repo packages it may import. A suite sees every layer below it and never
# another suite; core is shared by all three, so it may not reach up into pretrain either.
ALLOWED: dict[str, set[str]] = {
    "ibl_bwb_eval": set(),
    "core": {"ibl_bwb_eval"},
    "pretrain": {"ibl_bwb_eval", "core"},
    "ts1": {"ibl_bwb_eval", "core", "pretrain"},
    "ts2": {"ibl_bwb_eval", "core", "pretrain"},
    "ts3": {"ibl_bwb_eval", "core", "pretrain"},
}

PATHS = [p for pkg in ALLOWED for p in sorted((SRC / pkg).rglob("*.py"))]


def _violations(path: Path) -> set[tuple[str, str, int]]:
    """Every cross-layer import in one file, as (file, imported package, line)."""
    rel = path.relative_to(SRC).as_posix()
    pkg = rel.split("/")[0]
    found = set()
    for node in ast.walk(ast.parse(path.read_text())):
        if isinstance(node, ast.Import):
            modules = [alias.name for alias in node.names]
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            modules = [node.module]  # relative imports never leave the package
        else:
            continue
        for module in modules:
            root = module.split(".")[0]
            if root in ALLOWED and root != pkg and root not in ALLOWED[pkg]:
                found.add((rel, root, node.lineno))
    return found


@pytest.mark.parametrize("path", PATHS, ids=lambda p: p.relative_to(SRC).as_posix())
def test_module_imports_stay_within_its_layer(path):
    offenders = _violations(path)
    assert not offenders, "\n".join(f"{rel}:{line} imports {root}" for rel, root, line in offenders)
