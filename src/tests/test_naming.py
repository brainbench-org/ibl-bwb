"""A model has one rendering per layer, and the class layer keeps the paper's casing.

CONTRIBUTING.md's Naming section is the rule; this file holds the part a test can check.
INITIALISMS is the canonical spelling of every model name that is an initialism, so a
class called ``Nemo``, ``Poyo`` or ``Possm`` fails here even though it reads fine. Slugs
(``model=nemo_10M``, the model directory, the checkpoint filenames) are lowercase by the same
rule and are not class names, so they are not checked.

The suffix half of the rule: a pretrainer ends in ``Pretrain`` and a TS3 extractor in
``Extractor``. Renaming a *model* class is the one rename that reaches outside the repo,
since TS3 rebuilds a checkpoint's model from the ``_target_`` stored in it.
"""

import ast
import re
from pathlib import Path

import pytest

SRC = Path(__file__).resolve().parents[1]

INITIALISMS = [
    "NDT2",
    "NDT",
    "NEDS",
    "NEMO",
    "NuCLR",
    "POSSM",
    "POYO",
    "MtM",
    "LOLCAT",
    "LFADS",
    "CEBRA",
    "ISI",
    "RRR",
]

MODEL_DIRS = [SRC / "pretrain" / "models"] + [
    SRC / suite / "models" for suite in ("ts1", "ts2", "ts3")
]


def classes_in(path: Path) -> list[str]:
    tree = ast.parse(path.read_text())
    return [n.name for n in ast.walk(tree) if isinstance(n, ast.ClassDef)]


def model_files() -> list[Path]:
    return [
        p for root in MODEL_DIRS for p in sorted(root.rglob("*.py")) if "__pycache__" not in str(p)
    ]


@pytest.mark.parametrize("path", model_files(), ids=lambda p: str(p.relative_to(SRC)))
def test_class_names_use_the_paper_casing(path: Path):
    for name in classes_in(path):
        for initialism in INITIALISMS:
            wrong = re.match(initialism, name, re.IGNORECASE)
            if wrong and wrong.group(0) != initialism:
                pytest.fail(f"{path.relative_to(SRC)}: {name} should start with {initialism}")


@pytest.mark.parametrize("path", model_files(), ids=lambda p: str(p.relative_to(SRC)))
def test_role_suffixes(path: Path):
    rel = str(path.relative_to(SRC))
    for name in classes_in(path):
        if rel.startswith("pretrain/") and name.endswith("Trainer"):
            pytest.fail(f"{rel}: {name} is a pretrainer, so it ends in Pretrain")
        if rel.endswith("_extractor.py") and "Extractor" not in name:
            pytest.fail(f"{rel}: {name} lives in an extractor module but is not one")
