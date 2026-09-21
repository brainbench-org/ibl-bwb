"""A Hydra ``_target_`` names a package, so a file can move without breaking a checkpoint.

``_target_`` is the one identifier in the repo that outlives the checkout. A checkpoint
carries the config it was trained with, and ts3/models/base.py rebuilds the model from
that stored ``model._target_`` rather than from anything on disk today. A target that
reaches into a module therefore pins a filename forever: rename the file and every
checkpoint written before the rename stops loading.

So the module half of a target must be a package that exports the name, and the class
half must be in its ``__all__``. Everything inside the package is then free to move.
Resolution is static, like test_layering.py and test_naming.py, so this runs without the
training stack; ruff's F822 covers the other half, an ``__all__`` entry that does not
exist.

Third-party targets are skipped: only the repo's own packages are ours to arrange.
"""

import ast
import re
from pathlib import Path

import pytest
import yaml

SRC = Path(__file__).resolve().parents[1]
ROOTS = {"core", "ibl_bwb_eval", "pretrain", "ts1", "ts2", "ts3"}

# Values Hydra resolves at compose time (${...}) are not targets we can check statically.
INTERPOLATED = re.compile(r"\$\{")


def _targets(path: Path) -> list[str]:
    """Every ``_target_`` value in one config, at any nesting depth."""
    try:
        tree = yaml.safe_load(path.read_text())
    except yaml.YAMLError:  # a config the suite itself would fail to load
        return []

    found = []
    stack = [tree]
    while stack:
        node = stack.pop()
        if isinstance(node, dict):
            value = node.get("_target_")
            if isinstance(value, str) and not INTERPOLATED.search(value):
                found.append(value)
            stack.extend(node.values())
        elif isinstance(node, list):
            stack.extend(node)
    return found


def _exported_names(package: str) -> set[str] | None:
    """The ``__all__`` of a repo package, or None if that path is not a package."""
    init = SRC.joinpath(*package.split(".")) / "__init__.py"
    if not init.is_file():
        return None

    for node in ast.walk(ast.parse(init.read_text())):
        if (
            isinstance(node, ast.Assign)
            and any(isinstance(t, ast.Name) and t.id == "__all__" for t in node.targets)
            and isinstance(node.value, ast.List)
        ):
            return {e.value for e in node.value.elts if isinstance(e, ast.Constant)}
    return set()


CONFIGS = [
    p
    for p in sorted(SRC.rglob("configs/**/*.yaml"))
    if "outputs" not in p.parts  # hydra run dirs, snapshots of a config rather than one
]


@pytest.mark.parametrize("path", CONFIGS, ids=lambda p: p.relative_to(SRC).as_posix())
def test_targets_name_a_package_that_exports_them(path: Path):
    for target in _targets(path):
        package, _, name = target.rpartition(".")
        if package.split(".")[0] not in ROOTS:
            continue

        exported = _exported_names(package)
        assert exported is not None, (
            f"{target} names the module {package}, which pins a filename into every "
            f"checkpoint that stores this config. Target the package that exports it."
        )
        assert name in exported, f"{target}: {package} does not export {name} in __all__"
