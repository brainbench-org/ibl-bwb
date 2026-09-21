import datetime
import os
import sys
from pathlib import Path

from sphinx.util.typing import restify

sys.path.insert(0, os.path.abspath("."))
sys.path.insert(0, os.path.abspath("sphinxext"))
sys.path.insert(0, os.path.abspath("../../src"))

author = "The IBL BrainWideBench authors"  # matches the LICENSE copyright holder
project = "ibl-bwb"
version = "0.1.0"
copyright = f"{datetime.datetime.now().year}, {author}"


extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.intersphinx",
    "sphinx.ext.mathjax",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "myst_nb",
    "sphinx_autodoc_typehints",
    "sphinx_inline_tabs",
    "sphinx_copybutton",
    "sphinx_design",
    "sphinxcontrib.sass",
    "sphinxcontrib.bibtex",
    # see sphinxext/
    "autoshortsummary",
    "compact_bib",
]

bibtex_bibfiles = ["refs.bib"]
bibtex_default_style = "compact"
bibtex_reference_style = "author_year"

autosummary_generate = True

# The Docs workflow still publishes to GitHub Pages, but app-brain-wide-bench's nginx
# reverse-proxies that build under bwb.iblcore.org/docs/ — that's the address every page's
# canonical URL and the README's page links should carry, not the github.io one behind it.
html_baseurl = "https://bwb.iblcore.org/docs/"

html_theme = "pydata_sphinx_theme"
html_theme_options = {
    "logo": {
        "text": "IBL BWB",
        # The mark without the wordmark, which is illegible at navbar height and only
        # repeats the text beside it. Keyed to transparent, so one file serves both
        # the light and the dark navbar.
        "image_light": "_static/logo_mark.png",
        "image_dark": "_static/logo_mark.png",
    },
    # same order as the pill row on the landing page, which sits below this
    "icon_links": [
        {
            "name": "Leaderboard",
            "url": "https://bwb.iblcore.org",
            "icon": "fa-solid fa-ranking-star",
        },
        {
            "name": "GitHub",
            "url": "https://github.com/brainbench-org/ibl-bwb",
            "icon": "fa-brands fa-github",
        },
    ],
    "navbar_align": "left",
    # keep every top-level page in the navbar instead of a "More" dropdown
    "header_links_before_dropdown": 10,
    "navbar_start": ["navbar-logo"],
    "navbar_center": ["navbar-nav"],
    "navbar_end": ["theme-switcher", "navbar-icon-links"],
    # The landing page's only heading is Citation, so its page contents list holds a
    # single entry and earns none of the column it sits in.
    "secondary_sidebar_items": {
        "**": ["page-toc", "edit-this-page", "sourcelink"],
        "index": [],
    },
    "pygments_light_style": "a11y-light",
    "pygments_dark_style": "a11y-dark",
}

# The guides and the bibliography read top to bottom: the navbar reaches every one of
# them and the in-page contents cover the rest, so the section nav only repeats what is
# already on screen. The generated API pages keep the theme default, where the module
# list is the only way through them.
html_sidebars = {
    "guides/*": [],
    "references": [],
}

html_static_path = ["_static", "generated/css", "js"]
html_css_files = ["css/custom.css"]
html_js_files = ["scripts/sidebar-collapse.js"]
templates_path = ["_templates"]

add_module_names = True
autodoc_member_order = "bysource"

suppress_warnings = [
    "autodoc.import_object",
    "sphinx_autodoc_typehints.guarded_import",
]

intersphinx_mapping = {
    "python": ("https://docs.python.org/", None),
    "numpy": ("http://docs.scipy.org/doc/numpy", None),
    "h5py": ("http://docs.h5py.org/en/latest/", None),
    "torch": ("https://pytorch.org/docs/stable", None),
}

myst_enable_extensions = [
    "html_admonition",
    "html_image",
]

nb_execution_mode = "off"

# the mark alone: the wordmark is illegible at favicon sizes
html_favicon = "_static/favicon.png"
html_copy_source = False
html_show_sourcelink = True
# Compile scss files into css files using sphinxcontrib-sass
sass_src_dir, sass_out_dir = "scss", "generated/css/styles"
sass_targets = {
    f"{file.stem}.scss": f"{file.stem}.css" for file in Path(sass_src_dir).glob("*.scss")
}
Path("generated/css/").mkdir(exist_ok=True, parents=True)


from api_reference import build_api_rst

build_api_rst()


def add_js_css_files(app, pagename, templatename, context, doctree):
    """Load additional JS and CSS files only for certain pages.

    Note that `html_js_files` and `html_css_files` are included in all pages and
    should be used for the ones that are used by multiple pages. All page-specific
    JS and CSS files should be added here instead.
    """
    if pagename == "generated/api/all":
        # External: jQuery and DataTables
        app.add_js_file("https://code.jquery.com/jquery-3.7.0.js")
        app.add_js_file("https://cdn.datatables.net/2.0.0/js/dataTables.min.js")
        app.add_css_file("https://cdn.datatables.net/2.0.0/css/dataTables.dataTables.min.css")
        # Internal: API search initialization and styling
        app.add_js_file("scripts/api-search.js")
        app.add_css_file("styles/api-search.css")
    elif pagename.startswith("generated/api"):
        app.add_css_file("styles/api.css")


def _process_bases(app, name, obj, options, bases):
    # This shows torch.utils.data.dataset.Dataset as the base
    # without this, it would show up as "Dataset"
    bases[:] = [restify(b, "fully-qualified-except-typing") for b in obj.__bases__]


def setup(app):
    app.connect("autodoc-process-bases", _process_bases)
    # triggered just before the HTML for an individual page is created
    app.connect("html-page-context", add_js_css_files)
