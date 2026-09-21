"""Evaluation contract for the IBL BrainWideBench benchmark.

Everything an external consumer needs to produce a valid submission and to score one,
and nothing else: the task vocabulary and readout specs, the reported metrics, the
evaluation protocol (seeds, selection metric, eval sessions), the on-disk prediction
format, the per-suite scorers and the cross-suite ranking.

Only the stdlib-only names are re-exported here, because importing any submodule runs
this file first: :class:`ibl_bwb_eval.predictions.PredictionsWriter` needs torch, so it is
imported from its own module rather than from the package root.

Every readout name carries its suite, the way ``TS1Task`` and ``TS3Task`` do, so all three
suites' specs are exported side by side and no call site depends on context to say which
one it meant.

This package imports nothing from ``core``, ``pretrain`` or the task suites, and its
dependency set is the ``scoring`` extra: numpy, scipy, scikit-learn, torch,
torchmetrics, safetensors and rich. No torch_brain, hydra, ray or wandb. That
isolation is enforced by ``tests/test_ibl_bwb_eval_isolation.py``.
"""

from ibl_bwb_eval._version import __version__
from ibl_bwb_eval.protocol import EVAL_RECORDING_IDS, EVAL_SEEDS, SELECTION_METRIC, SWEEP_SEED
from ibl_bwb_eval.tasks import (
    COSMOS_LABELS,
    SUITE_TASKS,
    DataType,
    ReadoutSpec,
    TargetResolution,
    TS1ReadoutSpec,
    TS1Task,
    TS2ReadoutSpec,
    TS2Task,
    TS3ReadoutSpec,
    TS3Task,
    check_ts3_label_order,
    get_ts1_readout_spec,
    get_ts1_supported_tasks,
    get_ts2_readout_spec,
    get_ts3_readout_spec,
    is_task_of,
    task_id,
)

__all__ = [
    "COSMOS_LABELS",
    "EVAL_RECORDING_IDS",
    "EVAL_SEEDS",
    "SELECTION_METRIC",
    "SUITE_TASKS",
    "SWEEP_SEED",
    "DataType",
    "ReadoutSpec",
    "TS1ReadoutSpec",
    "TS1Task",
    "TS2ReadoutSpec",
    "TS2Task",
    "TS3ReadoutSpec",
    "TS3Task",
    "TargetResolution",
    "__version__",
    "check_ts3_label_order",
    "get_ts1_readout_spec",
    "get_ts1_supported_tasks",
    "get_ts2_readout_spec",
    "get_ts3_readout_spec",
    "is_task_of",
    "task_id",
]

__api_ref__ = {
    "description": None,
    "sections": [
        {
            "title": "Tasks",
            "autosummary": [
                "TS1Task",
                "TS2Task",
                "TS3Task",
                "task_id",
                "is_task_of",
                "get_ts1_supported_tasks",
            ],
        },
        {
            "title": "Readout types",
            "description": (
                "The suite-neutral pieces: the slice a model sizes its readout head from, "
                "and the two enums a spec describes its target with."
            ),
            "autosummary": ["ReadoutSpec", "TargetResolution", "DataType"],
        },
        {
            "title": "TS1 readouts",
            "autosummary": ["TS1ReadoutSpec", "get_ts1_readout_spec"],
        },
        {
            "title": "TS2 readouts",
            "description": (
                "TS2 sizes no readout head, so its spec names the axis the held-out mask "
                "varies along instead, and the metrics the task is scored with."
            ),
            "autosummary": ["TS2ReadoutSpec", "get_ts2_readout_spec"],
        },
        {
            "title": "TS3 readouts",
            "autosummary": [
                "TS3ReadoutSpec",
                "get_ts3_readout_spec",
                "COSMOS_LABELS",
                "check_ts3_label_order",
            ],
        },
    ],
}
