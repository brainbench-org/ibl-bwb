"""Evaluation-protocol constants, shared by every entry point that reports results.

``SWEEP_SEED`` is deliberately not in ``EVAL_SEEDS``: hyperparameters are selected on a
seed that never contributes to a reported number.
"""

from pathlib import Path

EVAL_SEEDS = [43, 44, 45, 46, 47]
SWEEP_SEED = 42
SELECTION_METRIC = "best/val/avg"

EVAL_RECORDING_IDS = Path(__file__).resolve().parent / "data" / "eval_recording_ids.txt"
