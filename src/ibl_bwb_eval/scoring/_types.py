from typing import TypeAlias

# Maps metric name -> (mean, sem, n). sem is None when n == 1.
MetricSummary: TypeAlias = dict[str, tuple[float, float | None, int]]

# Raw per-seed scores, keyed by either:
#   (label, task, recording_id, seed): suites with a session dimension (TS1/TS2's
#       score_dir() output already matches this shape).
#   (label, task, seed): suites with no session dimension (e.g. TS3,
#       a single atlas-level task run over the whole dataset).
# aggregation.aggregate() auto-detects which shape each key is and normalizes the
# 3-tuple form internally.
RawKey: TypeAlias = tuple[str, str, str, int] | tuple[str, str, int]
RawScores: TypeAlias = dict[RawKey, dict[str, float]]
