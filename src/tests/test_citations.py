"""Every model cites its paper the same way: one entry in refs.bib, named on its summary line.

The citation lives on the first line of the model class docstring, because that line is
what the API tables show, and it is a ``:cite:`` role rather than a URL so the reference
renders once in docs/source/references.rst and cannot drift between models. Reference
implementations are links, not bib entries, and sit in the paragraph below. Trainers,
extractors and eval wrappers never repeat a citation: they point at the model class.

MODEL_CITATIONS is the registry. A generic baseline (``Linear``, ``MLP``,
``GRU``, ``TCN``, the statistical baselines, the ISI extractor) is deliberately absent; add a model
here as soon as a paper exists for it. Markdown READMEs link the paper URL directly,
since roles do not render on GitHub.
"""

import ast
import re
from pathlib import Path

import pytest

SRC = Path(__file__).resolve().parents[1]
REFS = SRC.parent / "docs" / "source" / "refs.bib"

CITE = re.compile(r":cite:`([^`]+)`")
BIB_KEY = re.compile(r"@\w+\s*\{\s*([^,\s]+)\s*,")

# This file spells the role out, so its keys are text and not citations.
NOT_CITATIONS = {"tests/test_citations.py"}

# "<path under src>::<class>" -> the refs.bib key its summary line must carry.
MODEL_CITATIONS = {
    "pretrain/models/mtm/mtm.py::MtM": "mtm",
    "pretrain/models/ndt2/ndt2.py::NDT2": "ndt2",
    "pretrain/models/ndt_stitch/ndt_stitch.py::NDTStitch": "ndt",
    "pretrain/models/neds/neds.py::NEDS": "neds",
    "pretrain/models/nemo/nemo.py::NEMO": "nemo",
    "pretrain/models/nuclr/nuclr.py::NuCLR": "nuclr",
    "pretrain/models/possm/possm.py::POSSM": "possm",
    "pretrain/models/poyo/poyo.py::POYO": "poyo",
    "pretrain/models/poyo_plus/poyo_plus.py::POYOPlus": "poyo_plus",
    "pretrain/models/rrr/rrr.py::RRRDecoder": "rrr",
    "ts1/models/single_session/cebra/cebra.py::CEBRA": "cebra",
    "ts1/models/single_session/ndt_superv/ndt_superv.py::NDTSuperv": "ndt",
    "ts2/models/single_session/lfads/lfads.py::LFADS": "lfads",
    "ts2/models/single_session/ndt/ndt.py::NDT": "ndt",
    "ts3/models/supervised/lolcat/lolcat.py::LOLCAT": "lolcat",
}

PY_FILES = [p for p in sorted(SRC.rglob("*.py")) if str(p.relative_to(SRC)) not in NOT_CITATIONS]
DOC_FILES = sorted((SRC.parent / "docs" / "source").rglob("*.rst"))


def bib_keys() -> set[str]:
    return set(BIB_KEY.findall(REFS.read_text()))


def cited_keys(paths: list[Path]) -> set[str]:
    return {
        key.strip()
        for p in paths
        for group in CITE.findall(p.read_text())
        for key in group.split(",")
    }


def summary_line(path: Path, cls_name: str) -> str:
    tree = ast.parse((SRC / path).read_text())
    cls = next(
        (n for n in ast.walk(tree) if isinstance(n, ast.ClassDef) and n.name == cls_name), None
    )
    assert cls is not None, f"{path}: no class {cls_name}"
    doc = ast.get_docstring(cls)
    assert doc, f"{path}::{cls_name}: no docstring"
    return doc.strip().splitlines()[0]


@pytest.mark.parametrize("target,key", sorted(MODEL_CITATIONS.items()))
def test_model_summary_line_cites_its_paper(target: str, key: str):
    path, cls_name = target.split("::")
    line = summary_line(Path(path), cls_name)
    keys = {k.strip() for group in CITE.findall(line) for k in group.split(",")}
    assert key in keys, f"{target}: summary line must cite `{key}`, got: {line}"


def test_cited_keys_are_defined():
    undefined = cited_keys(PY_FILES + DOC_FILES) - bib_keys()
    assert not undefined, f"cited but missing from refs.bib: {sorted(undefined)}"


def test_bib_entries_are_cited():
    orphans = bib_keys() - cited_keys(PY_FILES + DOC_FILES)
    assert not orphans, f"in refs.bib but never cited: {sorted(orphans)}"
