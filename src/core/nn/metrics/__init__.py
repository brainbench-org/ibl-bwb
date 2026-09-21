"""Training-time diagnostics. The reported metrics live in :mod:`ibl_bwb_eval.metrics`."""

from .rankme import rankme

__all__ = ["rankme"]

__api_ref__ = {
    "description": None,
    "sections": [{"title": None, "autosummary": __all__}],
}
