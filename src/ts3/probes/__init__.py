"""The classifiers fit on a saved embeddings file, one embedding per unit.

Everything downstream of a probe belongs to ``ts3/eval.py``: the multi-unit readout, the
reports and the submission files. A new probe therefore implements
:meth:`Probe.fit_predict` and inherits the rest of the protocol.
"""

from ts3.probes.base import Probe
from ts3.probes.linear import LinearProbe
from ts3.probes.mlp import MLPProbe

__all__ = ["LinearProbe", "MLPProbe", "Probe"]

__api_ref__ = {
    "description": None,
    "sections": [
        {"title": None, "autosummary": ["Probe", "LinearProbe", "MLPProbe"]},
    ],
}
