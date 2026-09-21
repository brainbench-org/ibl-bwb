"""Score aggregation and ranking across all suites.

Turns the per-seed raw scores produced by ``ibl_bwb_eval.scoring.{ts1,ts2,ts3}.score_dir`` into
benchmark-style rankings, in two steps:

- :func:`aggregate` clips designated metrics (e.g. R^2/D^2/bps, which are unbounded
  below and null-relative: negative just means worse than the null model) at a
  floor *before* averaging over seeds, then groups by
  ``(label, task, recording_id)``, the same grouping ``summarize()`` uses in each
  suite, just with the clip applied first.
- :func:`rank` takes that per-recording, per-task summary and produces anchored,
  significance-based competition ranks (Welch's t-test step down sorted means, ties
  when not significantly different), averaged over recording_id within each task.

Usage:
    python -m ibl_bwb_eval.scoring.aggregation --pred_dir predictions/ --gt_dir ground_truth/
"""

import warnings
from collections import defaultdict
from collections.abc import Mapping
from pathlib import Path
from typing import Literal

import numpy as np
from rich.console import Console
from rich.table import Table
from safetensors import safe_open
from scipy import stats

from ibl_bwb_eval.scoring._types import MetricSummary, RawScores
from ibl_bwb_eval.tasks import is_task_of

__api_ref__ = {
    "description": None,
    "sections": [
        {
            "title": None,
            "description": (
                "``from_ts3`` first, since TS3's raw keys carry no session and the other "
                "two suites' already match what ``aggregate`` expects."
            ),
            "autosummary": ["from_ts3", "aggregate", "rank"],
        }
    ],
}

# Sentinel recording_id aggregate() assigns to session-less (3-tuple) raw keys.
NO_RECORDING_ID = "__all__"


def from_ts3(raw: dict[tuple[str, int], dict[str, float]], task: str) -> RawScores:
    """Inject ``task`` into ts3_scoring.score_dir's ``(label, seed)`` keys.

    ``ts3_scoring.score_dir`` doesn't retain ``task`` in its raw key (a scoring run
    only ever covers one TS3 task), so the caller supplies it here. The result is
    the session-less ``(label, task, seed)`` shape :func:`aggregate` recognizes and
    normalizes automatically.
    """
    return {(label, task, seed): metrics for (label, seed), metrics in raw.items()}


def _mean_sem_n(values: list[float]) -> tuple[float, float | None, int]:
    n = len(values)
    mean = float(np.mean(values))
    sem = float(np.std(values, ddof=1) / np.sqrt(n)) if n > 1 else None
    return mean, sem, n


def aggregate(
    raw: RawScores,
    clip_metrics: frozenset[str] = frozenset({"r2", "poisson_d2", "bps"}),
    clip_min: float = 0.0,
) -> dict[tuple[str, str, str], MetricSummary]:
    """Clip designated metrics at ``clip_min`` per seed, then aggregate over seeds.

    Accepts raw keys in either the 4-tuple ``(label, task, recording_id, seed)``
    shape or the 3-tuple ``(label, task, seed)`` shape for suites with no session
    dimension (e.g. TS3).

    Returns a dict keyed by ``(label, task, recording_id)`` where each value maps
    metric name to ``(mean, sem, n)``, the same shape as each suite's own
    ``summarize()``, with clipping applied.
    """
    grouped: dict[tuple[str, str, str], dict[str, list[float]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for key, metrics in raw.items():
        if len(key) == 4:
            label, task, recording_id, _seed = key
        elif len(key) == 3:
            label, task, _seed = key
            recording_id = NO_RECORDING_ID
        else:
            raise ValueError(f"raw key must be a 3- or 4-tuple, got {key!r}")

        for metric_name, value in metrics.items():
            if metric_name in clip_metrics:
                value = max(value, clip_min)
            grouped[(label, task, recording_id)][metric_name].append(value)

    return {
        key: {metric_name: _mean_sem_n(values) for metric_name, values in metric_vals.items()}
        for key, metric_vals in grouped.items()
    }


def _is_significantly_better(
    mean_a: float,
    sem_a: float | None,
    n_a: int,
    mean_b: float,
    sem_b: float | None,
    n_b: int,
    alpha: float,
) -> bool:
    """One-sided Welch's t-test: is A significantly greater than B?"""
    if mean_a <= mean_b:
        return False  # can't be "better" if mean isn't higher
    if sem_a is None or sem_b is None:
        # single-seed run on at least one side, no variance to test against.
        return True
    se = np.sqrt(sem_a**2 + sem_b**2)
    if se == 0:
        return True  # zero variance and mean_a > mean_b
    t = (mean_a - mean_b) / se
    # Welch-Satterthwaite df
    num = (sem_a**2 + sem_b**2) ** 2
    den = (sem_a**4) / (n_a - 1) + (sem_b**4) / (n_b - 1)
    df = num / den
    p = stats.t.sf(t, df)  # one-sided
    return p < alpha


RankMethod = Literal["step_down", "naive"]


def _assign_ranks_anchored(
    means: list[float],
    sems: list[float | None],
    ns: list[int],
    alpha: float,
    method: RankMethod = "step_down",
) -> list[int]:
    """Competition ranking (1,2,2,4), as ranks aligned with input order.

    Walks sorted-descending, comparing each candidate against the current anchor only.

    ``method="step_down"`` promotes to a new anchor only when it's significantly
    better than the current one (one-sided Welch's t-test at ``alpha``). Candidates
    that aren't significantly worse tie with the current anchor. ``method="naive"`` skips
    the test and promotes on any strictly higher mean, so ties only occur for exactly
    equal means.
    """
    if method == "naive":

        def is_better(mean_a, _sem_a, _n_a, mean_b, _sem_b, _n_b, _alpha):
            return mean_a > mean_b
    elif method == "step_down":
        is_better = _is_significantly_better
    else:
        raise ValueError(f"Unknown ranking method {method!r}, expected 'step_down' or 'naive'")

    order = sorted(range(len(means)), key=lambda i: -means[i])
    ranks: list[int] = [0] * len(means)
    rank = 1
    anchor = None
    num_tied = 0
    for i in order:
        if anchor is None:
            ranks[i] = rank
            anchor = i
        elif is_better(
            means[anchor],
            sems[anchor],
            ns[anchor],
            means[i],
            sems[i],
            ns[i],
            alpha,
        ):
            rank += num_tied + 1
            ranks[i] = rank
            anchor = i
            num_tied = 0
        else:
            num_tied += 1
            ranks[i] = rank
    return ranks


def rank(
    summary: dict[tuple[str, str, str], MetricSummary],
    primary_metric: Mapping[str, str],
    alpha: float = 0.05,
    method: RankMethod = "step_down",
) -> dict[str, dict[str, float]]:
    """Competition ranking, averaged over recording_id within each task.

    For each ``(task, recording_id)``, ranks labels by ``primary_metric[task]``.
    With ``method="step_down"`` (default), a candidate is only promoted past the
    current anchor if a one-sided Welch's t-test finds it significantly better at
    ``alpha``. Ties otherwise. With ``method="naive"``, ranking is by sorted mean
    alone, no significance test, ties only on exact equality. Ranks are then
    averaged over recording_id within each task. For TS3, whose rows all share a
    single recording_id sentinel, that average is a no-op over one value.

    Warns (``UserWarning``) per task where some recording_id has fewer than 2 labels
    scored on it (no comparison to make there, so that recording's rank is trivial).
    This typically means the labels being ranked weren't evaluated on the same
    sessions.

    Args:
        summary: output of :func:`aggregate`.
        primary_metric: task id -> metric name to rank that task on (e.g. from each
            suite's own readout spec / metric convention).
        alpha: significance threshold for the pairwise comparisons, used only when
            ``method="step_down"``.
        method: ``"step_down"`` or ``"naive"``, see above.

    Returns:
        Dict keyed by task, mapping label -> avg rank for that task. This is the
        row-oriented dict-of-dicts shape ``pd.DataFrame(result)`` expects directly
        (columns=task, index=label, NaN where a label has no score for a task);
        overall rank across tasks is then a simple ``.mean(axis=1)`` on that frame.
    """
    by_task: dict[str, dict[str, dict[str, tuple[float, float | None, int]]]] = defaultdict(
        lambda: defaultdict(dict)
    )
    for (label, task, recording_id), metrics in summary.items():
        metric_name = primary_metric[task]
        if metric_name not in metrics:
            raise KeyError(
                f"primary metric {metric_name!r} for task {task!r} not found for label "
                f"{label!r}, recording {recording_id!r}"
            )
        by_task[task][recording_id][label] = metrics[metric_name]

    avg_rank_per_task: dict[str, dict[str, float]] = {}
    for task, per_recording in by_task.items():
        incomplete = [rec for rec, per_label in per_recording.items() if len(per_label) < 2]
        if incomplete:
            shown = incomplete[:5]
            suffix = ", ..." if len(incomplete) > 5 else ""
            warnings.warn(
                f"rank(): task {task!r} has {len(incomplete)}/{len(per_recording)} recording_id(s) "
                f"with fewer than 2 labels scored ({shown}{suffix}) -- no comparison is possible "
                "there, so that recording's rank is trivial, not statistical. This usually means "
                "the labels being ranked weren't evaluated on the same sessions.",
                stacklevel=2,
            )

        per_label_ranks: dict[str, list[int]] = defaultdict(list)
        for per_label in per_recording.values():
            labels = list(per_label)
            means = [per_label[label][0] for label in labels]
            sems = [per_label[label][1] for label in labels]
            ns = [per_label[label][2] for label in labels]
            ranks = _assign_ranks_anchored(means, sems, ns, alpha, method)
            for label, r in zip(labels, ranks, strict=True):
                per_label_ranks[label].append(r)
        avg_rank_per_task[task] = {
            label: float(np.mean(rs)) for label, rs in per_label_ranks.items()
        }

    return avg_rank_per_task


def _discover_ts3_task(pred_dir: Path) -> str | None:
    """Find the TS3 task id from prediction file metadata, or None if there isn't one.

    ts3_scoring.score_dir doesn't retain the task in its raw key (it assumes a single
    TS3 task per scoring run), so main() recovers it here to pass to from_ts3.
    """
    for path in sorted(pred_dir.rglob("seed_*.safetensors")):
        with safe_open(str(path), framework="pt") as f:
            task = f.metadata()["task"]
        if is_task_of("ts3", task):
            return task
    return None


def aggregate_over_seeds(
    raw: RawScores,
    clip_metrics: frozenset[str] = frozenset({"r2", "poisson_d2", "bps"}),
    clip_min: float = 0.0,
) -> dict[tuple[str, str], MetricSummary]:
    """Aggregate scores over seeds, giving one value per seed.

    Given raw scores, this:

    1. averages over recording_id **within each seed**, after clipping ``clip_metrics``
       per (recording_id, seed), giving one value per seed;
    2. mean and SEM over those per-seed values, so ``sem`` is
       ``std(ddof=1) / sqrt(n_seeds)`` and ``n`` counts seeds, not recordings.

    Returns a dict keyed by ``(label, task)`` where each value maps metric name to
    ``(mean, sem, n)``, the same shape as each suite's own ``summarize()``, with clipping
    applied.

    Accepts either raw key shape :func:`aggregate` does; for a suite with no session
    dimension (TS3) stage 1 is a no-op and the per-seed values pass straight through.
    """
    per_seed: dict[tuple[str, str, str], dict[str, list[float]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for key, metrics in raw.items():
        if len(key) == 4:
            label, task, _recording_id, seed = key
        elif len(key) == 3:
            label, task, seed = key
        else:
            raise ValueError(f"raw key must be a 3- or 4-tuple, got {key!r}")

        for metric_name, value in metrics.items():
            if metric_name in clip_metrics:
                value = max(value, clip_min)
            per_seed[(label, task, metric_name)][seed].append(value)

    out: dict[tuple[str, str], MetricSummary] = defaultdict(dict)
    for (label, task, metric_name), by_seed in per_seed.items():
        seed_means = [float(np.mean(values)) for values in by_seed.values()]
        out[(label, task)][metric_name] = _mean_sem_n(seed_means)
    return dict(out)


def _by_task(
    scores: dict[tuple[str, str], MetricSummary], primary_metric: Mapping[str, str]
) -> dict[str, dict[str, tuple[float, float | None, int]]]:
    """Reshape :func:`aggregate_over_seeds` to ``task -> label -> entry``, primary metric only."""
    out: dict[str, dict[str, tuple[float, float | None, int]]] = defaultdict(dict)
    for (label, task), metrics in scores.items():
        metric_name = primary_metric.get(task)
        if metric_name is not None and metric_name in metrics:
            out[task][label] = metrics[metric_name]
    return dict(out)


def _by_task_metric(
    scores: dict[tuple[str, str], MetricSummary],
) -> dict[tuple[str, str], dict[str, tuple[float, float | None, int]]]:
    """Reshape :func:`aggregate_over_seeds` to ``(task, metric) -> label -> entry``."""
    out: dict[tuple[str, str], dict[str, tuple[float, float | None, int]]] = defaultdict(dict)
    for (label, task), metrics in scores.items():
        for metric_name, entry in metrics.items():
            out[(task, metric_name)][label] = entry
    return dict(out)


def print_aggregate_table(
    raw: RawScores,
    primary_metric: Mapping[str, str],
    console: Console | None = None,
) -> None:
    """Print the reported scores (mean ± SEM over seeds, see :func:`aggregate_over_seeds`).

    Uses the same label x task layout as :func:`print_rank_table`, showing the score each
    rank there is based on.
    """
    console = console or Console()
    scores = _by_task(aggregate_over_seeds(raw), primary_metric)

    tasks = sorted(scores)
    labels = sorted({label for per_task in scores.values() for label in per_task})

    table = Table(title="Aggregated Scores", show_header=True, show_lines=False)
    table.add_column("Label", style="magenta", no_wrap=True)
    for task in tasks:
        table.add_column(task, style="green", justify="right")

    for label in labels:
        cells = []
        for task in tasks:
            entry = scores[task].get(label)
            if entry is None:
                cells.append("[dim]--[/dim]")
            else:
                mean, sem, _n = entry
                sem_str = f"± {sem:.3f}" if sem is not None else "± --"
                cells.append(f"{mean:.3f} {sem_str}")
        table.add_row(label, *cells)

    console.print(table)


def print_all_metrics_table(
    raw: RawScores,
    console: Console | None = None,
) -> None:
    """Print every metric each task suite computes, not just the ranking one.

    Reports mean ± SEM over seeds (see :func:`aggregate_over_seeds`), one column per
    (task, metric).
    """
    console = console or Console()
    scores = _by_task_metric(aggregate_over_seeds(raw))

    columns = sorted(scores)
    labels = sorted({label for per_col in scores.values() for label in per_col})

    table = Table(title="All Aggregated Metrics", show_header=True, show_lines=False)
    table.add_column("Label", style="magenta", no_wrap=True)
    for task, metric in columns:
        table.add_column(f"{task}\n{metric}", style="green", justify="right")

    for label in labels:
        cells = []
        for col in columns:
            entry = scores[col].get(label)
            if entry is None:
                cells.append("[dim]--[/dim]")
            else:
                mean, sem, _n = entry
                sem_str = f"± {sem:.3f}" if sem is not None else "± --"
                cells.append(f"{mean:.3f} {sem_str}")
        table.add_row(label, *cells)

    console.print(table)


def print_rank_table(result: dict[str, dict[str, float]], console: Console | None = None) -> None:
    """Print a rank table, with rows=label and columns=task.

    The overall column is the mean rank across the tasks that label was scored on.
    """
    console = console or Console()
    tasks = sorted(result)
    labels = sorted({label for per_task in result.values() for label in per_task})

    table = Table(title="Benchmark Ranking", show_header=True, show_lines=False)
    table.add_column("Label", style="magenta", no_wrap=True)
    for task in tasks:
        table.add_column(task, style="green", justify="right")
    table.add_column("overall", style="bold cyan", justify="right")

    for label in labels:
        cells = []
        per_task_ranks = []
        for task in tasks:
            r = result[task].get(label)
            if r is None:
                cells.append("[dim]--[/dim]")
            else:
                cells.append(f"{r:.2f}")
                per_task_ranks.append(r)
        overall = f"{np.mean(per_task_ranks):.2f}" if per_task_ranks else "[dim]--[/dim]"
        table.add_row(label, *cells, overall)

    console.print(table)


def main():
    import argparse

    from ibl_bwb_eval.scoring import ts1 as ts1_scoring
    from ibl_bwb_eval.scoring import ts2 as ts2_scoring
    from ibl_bwb_eval.scoring import ts3 as ts3_scoring
    from ibl_bwb_eval.tasks import get_ts1_readout_spec

    parser = argparse.ArgumentParser(
        description="Aggregate TS1/TS2/TS3 scores and rank labels against ground truth.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--pred_dir", type=str, required=True, help="Root predictions directory.")
    parser.add_argument("--gt_dir", type=str, required=True, help="Root ground truth directory.")
    parser.add_argument(
        "--method", choices=["step_down", "naive"], default="step_down", help="Ranking method."
    )
    parser.add_argument(
        "--alpha", type=float, default=0.05, help="Significance threshold (step_down only)."
    )
    parser.add_argument(
        "--all-metrics",
        action="store_true",
        help="Also print every metric each task suite computes, not just the primary one used for ranking.",
    )
    args = parser.parse_args()

    pred_dir, gt_dir = Path(args.pred_dir), Path(args.gt_dir)
    console = Console()

    raw: RawScores = {}
    raw.update(ts1_scoring.score_dir(pred_dir, gt_dir))
    raw.update(ts2_scoring.score_dir(pred_dir, gt_dir))

    ts3_task = _discover_ts3_task(pred_dir)
    if ts3_task is not None:
        raw.update(from_ts3(ts3_scoring.score_dir(pred_dir, gt_dir), task=ts3_task))

    if not raw:
        console.print(f"[yellow]No predictions found under {pred_dir}[/yellow]")
        return

    summary = aggregate(raw)

    primary_metric: dict[str, str] = {}
    for _label, task, _recording_id in summary:
        if task in primary_metric:
            continue
        if is_task_of("ts1", task):
            primary_metric[task] = get_ts1_readout_spec(task.split("-", 1)[1]).primary_metric
        elif is_task_of("ts2", task):
            primary_metric[task] = "poisson_d2"
        elif task == ts3_task:
            primary_metric[task] = "macro/f1-score"

    print_aggregate_table(raw, primary_metric, console)
    if args.all_metrics:
        print_all_metrics_table(raw, console)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = rank(summary, primary_metric, alpha=args.alpha, method=args.method)
    for w in caught:
        console.print(f"[yellow]{w.message}[/yellow]")

    print_rank_table(result, console)


if __name__ == "__main__":
    main()
