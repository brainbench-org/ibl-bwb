"""Tests for the citation metadata: one source, and a sweep that cannot half-land.

The paper's title, authors, year and URL live in four files at once, so the failure this
guards against is a half-finished edit: changing three of them leaves the fourth quietly
advertising the old text to anyone who clicks "Cite this repository", and nothing else in
the repo would notice.

``CITATION.bib`` is the source. The docs include it, the README quotes it, and
``CITATION.cff`` restates it in the structured form GitHub reads.
"""

import re
from pathlib import Path

import pytest
import yaml

REPO = Path(__file__).resolve().parents[2]
BIB = REPO / "CITATION.bib"
CFF = REPO / "CITATION.cff"
README = REPO / "README.md"
INDEX = REPO / "docs" / "source" / "index.rst"
REFS = REPO / "docs" / "source" / "refs.bib"

# Every file the sweep has to touch, so a half-filled citation fails loudly.
SWEPT = (BIB, CFF, README, INDEX, REFS)

# The marker a placeholder leaves behind. Named rather than written out below, so that
# grepping the repo for unfinished work does not keep finding this file.
PLACEHOLDER = "TODO"


def _entries(text: str) -> dict[str, str]:
    """The bibtex entries in ``text``, keyed by cite key."""
    return {m.group(2): m.group(0) for m in re.finditer(r"@(\w+)\{([^,]+),.*?\n\}", text, re.S)}


def _fields(entry: str) -> dict[str, str]:
    """``key = {value}`` pairs, one per line, which is how CITATION.bib is written."""
    out = {}
    for line in entry.splitlines()[1:]:
        key, sep, value = line.partition("=")
        if sep:
            out[key.strip()] = value.strip().rstrip(",").strip().strip("{}").strip()
    return out


@pytest.fixture(scope="module")
def bib() -> dict[str, dict[str, str]]:
    # Either cite key could be renamed, so the markers say which entry is which.
    # They are the same markers the docs include by, so this cannot drift from the page.
    text = BIB.read_text()
    sections = {}
    for role in ("paper", "dataset"):
        after = text.split(f"%%% {role}\n", 1)
        assert len(after) == 2, f"CITATION.bib has no '%%% {role}' marker"
        entries = _entries(after[1].split("%%% ", 1)[0])
        assert len(entries) == 1, f"expected one entry under '%%% {role}', found {len(entries)}"
        sections[role] = _fields(next(iter(entries.values())))
    return sections


def test_the_readme_quotes_the_source_verbatim():
    """The README cannot include a file, so it copies; the copy has to be exact."""
    readme = README.read_text()
    for key, entry in _entries(BIB.read_text()).items():
        assert entry in readme, (
            f"README.md's {key} entry has drifted from CITATION.bib. Copy the entry "
            f"across rather than editing it in place."
        )


def test_the_docs_include_the_source_rather_than_copying_it():
    index = INDEX.read_text()
    assert index.count("literalinclude:: ../../CITATION.bib") == 2, (
        "docs/source/index.rst should include CITATION.bib once per entry"
    )
    assert "code-block:: bibtex" not in index, (
        "docs/source/index.rst has gone back to pasting bibtex; include the file instead"
    )


def test_the_docs_include_markers_still_exist():
    """``:start-after:`` fails silently in some Sphinx versions if the marker is gone."""
    bib = BIB.read_text()
    for marker in ("%%% paper", "%%% dataset"):
        assert bib.count(marker) == 1, (
            f"CITATION.bib needs exactly one {marker!r}, the marker index.rst includes "
            f"by; a second occurrence, even inside a comment, silently moves where the "
            f"include starts. Found {bib.count(marker)}."
        )


def test_the_cff_agrees_with_the_source(bib):
    """``preferred-citation`` is what citation tooling shows, so it is the copy that
    matters most and the one furthest from the bibtex people read."""
    cff = yaml.safe_load(CFF.read_text())
    paper, cited = bib["paper"], cff["preferred-citation"]
    # Only the fields both formats carry: a bibtex entry's shape depends on what it is,
    # so booktitle is there for a proceedings paper and absent for a preprint.
    for cff_key, bib_key in (("title", "title"), ("year", "year"), ("url", "url")):
        assert str(cited[cff_key]) == paper[bib_key], (
            f"CITATION.cff preferred-citation.{cff_key} says {cited[cff_key]!r}, "
            f"CITATION.bib says {paper[bib_key]!r}"
        )

    listed = [str(entry["value"]) for entry in cited.get("identifiers", [])]
    if "eprint" in paper:
        assert f"arXiv:{paper['eprint']}" in listed, (
            f"CITATION.bib is eprint {paper['eprint']}, CITATION.cff identifiers say {listed}"
        )

    dataset = bib["dataset"]
    titles = [str(ref.get("title")) for ref in cff.get("references", [])]
    assert dataset["title"] in titles, (
        f"CITATION.cff references do not carry the dataset title {dataset['title']!r}"
    )


def test_the_citation_is_filled_in_everywhere_or_nowhere():
    """Either the paper is still unpublished, or every copy of it says so no longer."""
    if PLACEHOLDER in BIB.read_text():
        pytest.skip("the paper is still unpublished, so placeholders are expected")

    stale = [path.relative_to(REPO) for path in SWEPT if PLACEHOLDER in path.read_text()]
    assert not stale, f"CITATION.bib is filled in but these still carry a placeholder: {stale}"
    buttons = [
        path.relative_to(REPO)
        for path in (README, INDEX)
        if f'href="{PLACEHOLDER}"' in path.read_text()
    ]
    assert not buttons, f"the paper button still points at the placeholder in {buttons}"
