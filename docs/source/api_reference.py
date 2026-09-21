"""Renders the API reference pages from each module's ``__api_ref__``.

``API_GROUPS`` below is the whole configuration: the cards the landing page shows, in
order, and the modules on each. A module reaches a page only by being listed there, which
``tests/test_documented_names.py`` enforces in both directions.

A module says what its own page holds in ``__api_ref__``:

description
    Prose under the module docstring, before the first section. ``None`` for most.
sections
    One or more ``{title, description, autosummary}``. ``title`` is the section heading,
    ``None`` to let the first section follow the module docstring with no heading of its
    own. ``description`` is optional prose under that heading. ``autosummary`` lists the
    names, which must be attributes of the module itself: a dotted path does not resolve.

So a page renders as:

|---------------------------------------------------------------------------------|
|     {{ module }}                                                                 |
|     ============                                                                 |
|     {{ module docstring }}                                                        |
|     {{ description }}                                                             |
|                                                                                   |
|     {{ section title }}      <---- omitted when the title is None, so the first   |
|     -----------------              section follows the docstring directly         |
|     {{ section description }}                                                     |
|     {{ section autosummary }}                                                     |
|                                                                                   |
|     More sections...                                                              |
|---------------------------------------------------------------------------------|

Cross-reference a page with ``:mod:`core.dataset```, and an object on one with
``:class:`~core.dataset.IBLBrainWideBench2026```. The pages carry no hand-written labels
for ``:ref:`` to target.

The landing page takes each module's card row from the first line of its docstring, so a
module listed here needs one.
"""

from importlib import import_module

import jinja2

# The API reference, grouped as the landing page presents it. Each group is a card;
# each module inside it is a row on that card, described by its own docstring.
API_GROUPS = [
    {
        "title": "Evaluation contract",
        "description": "What a submission must contain.",
        "modules": [
            "ibl_bwb_eval",
            "ibl_bwb_eval.metrics",
            "ibl_bwb_eval.multi_unit",
            "ibl_bwb_eval.predictions",
            "ibl_bwb_eval.entity_ids",
        ],
    },
    {
        "title": "Scoring",
        "description": "How a directory of submissions becomes the reported numbers.",
        "modules": [
            "ibl_bwb_eval.scoring.ts1",
            "ibl_bwb_eval.scoring.ts2",
            "ibl_bwb_eval.scoring.ts3",
            "ibl_bwb_eval.scoring.aggregation",
        ],
    },
    {
        "title": "Core",
        "description": "The pieces every task suite builds on.",
        "modules": [
            "core.data",
            "core.dataset",
            "core.finetuning",
            "core.model",
            "core.trainer",
            "core.samplers",
            "core.transforms",
            "core.nn",
            "core.nn.loss",
            "core.nn.metrics",
        ],
    },
    {
        "title": "Pretraining",
        "description": "Encoders pretrained once, then evaluated by each suite.",
        "modules": ["pretrain.models", "pretrain.datasets"],
    },
    {
        "title": "TS1: Behavior Prediction",
        "description": "Decoding behavior from neural population activity.",
        "modules": ["ts1", "ts1.models.single_session", "ts1.models.pretrained"],
    },
    {
        "title": "TS2: Neural Activity Prediction",
        "description": "Predicting activity across time and across neurons.",
        "modules": ["ts2", "ts2.models.single_session", "ts2.models.pretrained"],
    },
    {
        "title": "TS3: Brain Region Prediction",
        "description": "Predicting the region a single neuron was recorded in.",
        "modules": [
            "ts3",
            "ts3.models.inductive",
            "ts3.models.transductive",
            "ts3.models.supervised",
            "ts3.probes",
        ],
    },
]

# Modules to include in API reference, in landing-page order.
API_MODS = [module for group in API_GROUPS for module in group["modules"]]

API_REFERENCE = {m: import_module(m).__api_ref__ for m in API_MODS}


def _short_summary(module: str) -> str:
    """First line of the module docstring, or an empty string if it has none."""
    doc = import_module(module).__doc__ or ""
    return doc.strip().split("\n", 1)[0]


# API_GROUPS with each module paired with its short summary, for the landing page.
API_CARDS = [
    {
        **group,
        "entries": [{"module": m, "summary": _short_summary(m)} for m in group["modules"]],
    }
    for group in API_GROUPS
]


def build_api_rst():
    import pathlib
    import shutil

    generated = pathlib.Path(__file__).parent / "generated"
    generated.mkdir(exist_ok=True)

    # Clear any stale output first: a renamed or removed module otherwise leaves its old
    # generated/autosummary .rst files behind indefinitely, since the loop below only
    # writes files for what's currently in API_REFERENCE, never deletes orphans.
    shutil.rmtree(generated / "api", ignore_errors=True)
    (generated / "api").mkdir(exist_ok=True)

    # rst_templates
    # kwargs: args to pass to jinja
    rst_templates: list[dict] = [
        {
            "template_path": "api/index.rst.template",
            "target_path": "generated/api/index.rst",
            "kwargs": {"API_CARDS": API_CARDS},
        },
        {
            "template_path": "api/all.rst.template",
            "target_path": "generated/api/all.rst",
            "kwargs": {"API_REFERENCE": API_REFERENCE.items()},
        },
    ]

    for module in API_REFERENCE:
        rst_templates.append(
            {
                "template_path": "api/module.rst.template",
                "target_path": f"generated/api/{module}.rst",
                "kwargs": {"module": module, "module_info": API_REFERENCE[module]},
            }
        )

    for template in rst_templates:
        # Read the corresponding template file into jinja2
        with open(template["template_path"]) as f:
            t = jinja2.Template(f.read())

        # Render the template and write to the target
        with open(template["target_path"], "w") as f:
            f.write(t.render(**template["kwargs"]))
