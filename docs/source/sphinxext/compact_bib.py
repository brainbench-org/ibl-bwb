"""A compact bibliography style: authors, year, linked title, short venue, and nothing else.

pybtex's ``unsrt`` prints every author, the venue's full ceremonial name, then the URL and
the DOI, so a page whose job is to say which paper a model comes from reads as three lines
of noise per entry. This trims each entry to what a reader scans for and hangs the link on
the title, so no raw URL is shown.

Registered as the ``compact`` pybtex style; ``conf.py`` selects it with
``bibtex_default_style``.
"""

import re

from pybtex.plugin import register_plugin
from pybtex.richtext import Text
from pybtex.style.formatting import toplevel
from pybtex.style.formatting.unsrt import Style as UnsrtStyle
from pybtex.style.labels import BaseLabelStyle
from pybtex.style.template import field, first_of, href, join, node, optional, tag

# Conference names are ceremonial in the entries the publishers hand out ("Thirty-seventh
# Conference on ..."), and the year is already printed next to the authors.
VENUES = [
    (r"Neural Information Processing Systems.*Datasets and Benchmarks", "NeurIPS D&B"),
    (r"Neural Information Processing Systems", "NeurIPS"),
    (r"International Conference on Machine Learning", "ICML"),
    (r"International Conference on Learning Representations", "ICLR"),
    (r"Empirical Methods in Natural Language Processing", "EMNLP"),
    (r"International Conference on Spoken Language Translation", "IWSLT"),
    (r"arXiv preprint (arXiv:\S+)", r"\1"),
]


def short_venue(text) -> str:
    text = str(text)  # pybtex hands the field in as rich text
    for pattern, short in VENUES:
        if re.search(pattern, text, re.IGNORECASE):
            return re.sub(f".*{pattern}.*", short, text, flags=re.IGNORECASE)
    return text


@node
def compact_names(children, context, role, **kwargs):
    """Last names only: one, two joined by "and", or the first followed by "et al."."""
    persons = context["entry"].persons[role]
    names = [" ".join(p.last_names) or " ".join(p.first_names) for p in persons]
    if len(names) == 1:
        return Text(names[0])
    if len(names) == 2:
        return Text(f"{names[0]} and {names[1]}")
    return Text(f"{names[0]} et al.")


class KeyLabelStyle(BaseLabelStyle):
    """Label an entry with its citation key, which here is the model's own name."""

    def format_labels(self, sorted_entries):
        for entry in sorted_entries:
            yield entry.key


class CompactStyle(UnsrtStyle):
    default_label_style = "key"
    default_sorting_style = "author_year_title"

    def _linked_title(self):
        title = tag("em")[field("title")]
        return first_of[
            optional[href[field("url"), title]],
            optional[href[join["https://doi.org/", field("doi")], title]],
            title,
        ]

    def _entry(self, venue_field: str | None):
        return toplevel[
            join(sep=" ")[
                join[compact_names("author"), optional[" (", field("year"), ")"], "."],
                join[self._linked_title(), "."],
                optional[join[field(venue_field, apply_func=short_venue), "."]]
                if venue_field
                else "",
            ]
        ]

    def get_article_template(self, e):
        return self._entry("journal")

    def get_inproceedings_template(self, e):
        return self._entry("booktitle")

    def get_incollection_template(self, e):
        return self._entry("booktitle")

    def get_inbook_template(self, e):
        return self._entry("booktitle")

    def get_book_template(self, e):
        return self._entry("publisher")

    def get_techreport_template(self, e):
        return self._entry("institution")

    def get_phdthesis_template(self, e):
        return self._entry("school")

    def get_misc_template(self, e):
        return self._entry("howpublished")


register_plugin("pybtex.style.labels", "key", KeyLabelStyle)
register_plugin("pybtex.style.formatting", "compact", CompactStyle)


def setup(app):
    return {"version": "1.0", "parallel_read_safe": True, "parallel_write_safe": True}
