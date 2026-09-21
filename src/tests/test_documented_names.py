"""Documentation that nothing checks goes stale, or goes unread. Both are checked here.

A repo class named in prose has to be a class that exists. A ``_target_`` spelled out in
a guide, or a ``:class:`` reference in a docstring, is checked by nothing: Sphinx renders
an unresolved reference as plain text and the build still passes, so a rename leaves the
wrong name sitting in the docs until a reader tries it. test_config_targets.py covers the
same names inside configs; this covers every other place they are written down.

An ``__api_ref__`` has to reach a page. Only the modules in ``API_GROUPS`` are rendered,
so a block on any other module is inert: ten of them had accumulated, each listing names
their parent page already showed. Editing one looks like editing the docs and is not.

Anything shaped like a repo-qualified dotted path ending in a capitalised name is
checked, wherever it appears: guides, READMEs, and the module, class and function
docstrings under ``src/``. The last segment has to be capitalised, so a bare module path
(``pretrain.models.nemo.encoding``) is prose about a module rather than a claim that a
class exists, and is left alone.

Unlike test_config_targets.py this one imports, since only an import can say whether an
attribute is really there. It therefore needs the training stack, which is what the
``train`` CI job installs.
"""

import ast
import re
from importlib import import_module
from pathlib import Path

import pytest

SRC = Path(__file__).resolve().parents[1]
REPO = SRC.parent

ROOTS = ("core", "ibl_bwb_eval", "pretrain", "ts1", "ts2", "ts3")
DOTTED = re.compile(rf"\b(?:{'|'.join(ROOTS)})(?:\.\w+)+\.[A-Z]\w*")

PROSE = sorted(
    p
    for p in [
        *(REPO / "docs" / "source").rglob("*.rst"),
        *(REPO / "docs" / "source").rglob("*.md"),
        *SRC.rglob("*.md"),
        REPO / "README.md",
        REPO / "CONTRIBUTING.md",
    ]
    # docs/source/generated is written by the build from the modules themselves
    if p.is_file() and "generated" not in p.parts
)
PYTHON = sorted(p for p in SRC.rglob("*.py") if "__pycache__" not in p.parts)


def _docstrings(path: Path) -> str:
    nodes = (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)
    tree = ast.parse(path.read_text())
    return "\n".join(ast.get_docstring(n) or "" for n in ast.walk(tree) if isinstance(n, nodes))


def _check(text: str, where: Path) -> None:
    for dotted in sorted(set(DOTTED.findall(text))):
        module, _, name = dotted.rpartition(".")
        try:
            found = hasattr(import_module(module), name)
        except ImportError:
            found = False
        assert found, f"{where.relative_to(REPO)} names {dotted}, which does not exist"


@pytest.mark.parametrize("path", PROSE, ids=lambda p: p.relative_to(REPO).as_posix())
def test_names_written_in_prose_exist(path: Path):
    _check(path.read_text(), path)


@pytest.mark.parametrize("path", PYTHON, ids=lambda p: p.relative_to(SRC).as_posix())
def test_names_written_in_docstrings_exist(path: Path):
    _check(_docstrings(path), path)


def _rendered_modules() -> set[str]:
    """The modules ``API_GROUPS`` renders, read without importing it.

    api_reference.py pulls in jinja2, which only the ``docs`` extra installs.
    """
    tree = ast.parse((REPO / "docs" / "source" / "api_reference.py").read_text())
    groups = next(
        node.value
        for node in ast.walk(tree)
        if isinstance(node, ast.Assign)
        and any(getattr(t, "id", None) == "API_GROUPS" for t in node.targets)
    )
    return {
        module.value
        for group in groups.elts
        for key, value in zip(group.keys, group.values, strict=True)
        if key.value == "modules"
        for module in value.elts
    }


def test_every_api_ref_block_reaches_a_page():
    rendered = _rendered_modules()
    inert = sorted(
        path.parent.relative_to(SRC).as_posix().replace("/", ".")
        for path in SRC.rglob("__init__.py")
        if "__pycache__" not in path.parts and "__api_ref__" in path.read_text()
    )
    orphans = [module for module in inert if module not in rendered]
    assert not orphans, (
        "these carry an __api_ref__ that no page renders, because API_GROUPS does not "
        "list them: " + ", ".join(orphans)
    )


def test_every_api_ref_name_exists():
    """A name listed on a page has to be a name its module exports.

    autosummary resolves these itself, and a stale entry only warns, so a rename leaves
    one pointing at nothing until a reader clicks it. Pages are free to list less than
    ``__all__``: constants and type aliases are exported but have nothing to render.
    """
    for module in sorted(_rendered_modules()):
        mod = import_module(module)
        listed = [
            name for section in mod.__api_ref__["sections"] for name in section["autosummary"]
        ]
        missing = [name for name in listed if not hasattr(mod, name)]
        assert not missing, f"{module} lists {missing}, which it does not export"
        repeated = sorted({name for name in listed if listed.count(name) > 1})
        assert not repeated, f"{module} lists {repeated} more than once"
