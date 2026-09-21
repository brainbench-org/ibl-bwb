import sys

from sphinx.ext.autodoc import ModuleLevelDocumenter
from sphinx.pycode import ModuleAnalyzer, PycodeError


def _attr_doc(modname: str, name: str) -> list[str]:
    """The source-level docstring of a module attribute, empty when it has none.

    A module-level assignment has no ``__doc__`` to read, so the only record of its
    docstring is the source. ``TypeAlias = Literal[...]`` is the case here.
    """
    try:
        analyzer = ModuleAnalyzer.for_module(modname)
        analyzer.analyze()
    except PycodeError:
        return []
    return list(analyzer.find_attr_docs().get(("", name), []))


def _reexported_attr_doc(modname: str, name: str) -> list[str]:
    """``_attr_doc``, following a name a package re-exports from one of its submodules.

    The task aliases are written in ``ibl_bwb_eval.tasks.*`` and read from
    ``ibl_bwb_eval``, and an alias carries no ``__module__`` pointing home, so the
    submodule that holds the same object is the only way back to the source.
    """
    if doc := _attr_doc(modname, name):
        return doc
    obj = getattr(sys.modules.get(modname), name, None)
    if obj is None:
        return []
    prefix = modname + "."
    for submodule, module in list(sys.modules.items()):
        if (
            submodule.startswith(prefix)
            and getattr(module, name, None) is obj
            and (doc := _attr_doc(submodule, name))
        ):
            return doc
    return []


class ShortSummaryDocumenter(ModuleLevelDocumenter):
    """An autodocumenter that only renders the short summary of the object."""

    # Defines the usage: .. autoshortsummary:: {{ object }}
    objtype = "shortsummary"

    # Disable content indentation
    content_indent = ""

    # Avoid being selected as the default documenter for some objects, because we are
    # returning `can_document_member` as True for all objects
    priority = -99

    @classmethod
    def can_document_member(cls, member, membername, isattr, parent):
        """Allow documenting any object."""
        return True

    def get_object_members(self, want_all):
        """Document no members."""
        return (False, [])

    def add_directive_header(self, sig):
        """Override default behavior to add no directive header or options."""
        pass

    def add_content(self, more_content):
        """Override default behavior to add only the first line of the docstring.

        Modified based on the part of processing docstrings in the original
        implementation of this method.

        https://github.com/sphinx-doc/sphinx/blob/faa33a53a389f6f8bc1f6ae97d6015fa92393c4a/sphinx/ext/autodoc/__init__.py#L609-L622
        """
        sourcename = self.get_sourcename()
        docstrings = self.get_doc()

        # get_doc() returns [[]] for an object with no __doc__, so this asks whether
        # any real content came back before falling back to the source.
        if not any(line for block in docstrings or [] for line in block):
            docstrings = [_reexported_attr_doc(self.modname, ".".join(self.objpath))]

        if docstrings is not None:
            if not docstrings:
                docstrings.append([])
            # Get the first non-empty line of the processed docstring; this could lead
            # to unexpected results if the object does not have a short summary line.
            short_summary = next((s for s in self.process_doc(docstrings) if s), "<no summary>")
            self.add_line(short_summary, sourcename, 0)


def setup(app):
    app.add_autodocumenter(ShortSummaryDocumenter)
