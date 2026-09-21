"""Every version anyone else can hold has to be described somewhere.

The brainset version is stamped into each ``.h5`` and the eval version into each
prediction file, and both are the only handle their reader has on what they were given.
A bump without an entry leaves that reader with a number and no way to interpret it, so
the bump and the entry are pinned together here.

Only a version someone outside the project can hold needs describing. The brainset is
exempt below 0.1.0, its first public release, because what came before was internal and
was never distributed; the exemption retires itself the moment the literal reaches 0.1.0.
The eval package has no such era left, since it is already on PyPI, so every version of
it needs a section.
"""

import ast
import re
from pathlib import Path

from ibl_bwb_eval._version import __version__

ROOT = Path(__file__).parents[2]
PIPELINE = ROOT / "ibl_brain_wide_bench_2026" / "pipeline.py"
BRAINSET_CHANGELOG = ROOT / "ibl_brain_wide_bench_2026" / "CHANGELOG.md"
EVAL_CHANGELOG = ROOT / "packaging" / "ibl-bwb-eval" / "CHANGELOG.md"

FIRST_PUBLIC_BRAINSET = (0, 1, 0)


def _pipeline_derived_version() -> str:
    """The derived_version literal the pipeline stamps on every recording."""
    for node in ast.walk(ast.parse(PIPELINE.read_text())):
        if isinstance(node, ast.keyword) and node.arg == "derived_version":
            return node.value.value
    raise AssertionError(f"derived_version not found in {PIPELINE.name}")


def _has_heading(changelog: Path, title: str) -> bool:
    pattern = rf"^## {re.escape(title)}\b"
    return re.search(pattern, changelog.read_text(), re.MULTILINE) is not None


def test_brainset_version_has_a_changelog_entry():
    version = _pipeline_derived_version()
    if tuple(int(part) for part in version.split(".")) < FIRST_PUBLIC_BRAINSET:
        assert _has_heading(BRAINSET_CHANGELOG, "Unreleased"), (
            f"pipeline.py builds {version}, before the first public build, so "
            f"{BRAINSET_CHANGELOG.name} needs an Unreleased section to collect what that "
            "build will carry."
        )
        return
    assert _has_heading(BRAINSET_CHANGELOG, version), (
        f"pipeline.py builds {version}, which has no '## {version}' section in "
        f"{BRAINSET_CHANGELOG.name}. The bucket is rebuilt in place, so that section is "
        "all a downloader has to tell the builds apart."
    )


def test_eval_version_has_a_changelog_entry():
    assert _has_heading(EVAL_CHANGELOG, __version__), (
        f"ibl-bwb-eval is at {__version__}, which has no '## {__version__}' section in "
        f"{EVAL_CHANGELOG.name}. Rename the Unreleased section when bumping _version.py: "
        "a PyPI version is frozen once published."
    )
