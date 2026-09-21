"""Single source of truth for the package version.

Read by ``packaging/ibl-bwb-eval/pyproject.toml`` so the built wheel cannot drift from
it, and stamped into every prediction file's metadata as ``ibl_bwb_eval_version``.
"""

__version__ = "0.1.0"
