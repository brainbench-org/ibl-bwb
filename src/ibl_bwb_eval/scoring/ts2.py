"""TS2 scoring utility.

Loads prediction .safetensors files, aligns them to pre-generated ground truth files
by ``window_timestamps``, and computes Poisson D² and bits-per-spike.

Usage:
    python -m ibl_bwb_eval.scoring.ts2 --pred_dir predictions/<label> --gt_dir ground_truth/ts2
    python -m ibl_bwb_eval.scoring.ts2 --pred_dir predictions/<label>/my-model/ts2-co_smoothing --gt_dir ground_truth/ts2
"""

import argparse
from collections import defaultdict
from pathlib import Path

import numpy as np
from rich.console import Console
from rich.table import Table
from safetensors import safe_open
from safetensors.torch import load_file

from ibl_bwb_eval.metrics import BPS, PoissonD2Score
from ibl_bwb_eval.scoring._types import MetricSummary
from ibl_bwb_eval.tasks import is_task_of

__api_ref__ = {
    "description": None,
    "sections": [{"title": None, "autosummary": ["score_file", "score_dir", "summarize"]}],
}


def score_file(pred_path: Path, gt_path: Path) -> dict[str, float]:
    """Score one prediction file against its ground truth file.

    Predictions and ground truth are aligned by ``window_timestamps``.

    Args:
        pred_path: Path to a ``seed_N.safetensors`` prediction file.
        gt_path: Path to the matching ``ground_truth.safetensors`` file.

    Returns:
        Dict mapping metric name to scalar value.
    """
    pred = load_file(str(pred_path))
    gt = load_file(str(gt_path))

    pred_ts = pred["window_timestamps"].numpy()  # (B_total,)
    gt_ts = gt["window_timestamps"].numpy()  # (B_total,)
    _, pred_idx, gt_idx = np.intersect1d(pred_ts, gt_ts, return_indices=True)

    if len(pred_idx) == 0:
        raise ValueError("No overlapping windows between predictions and ground truth.")

    predictions = pred["predictions"][pred_idx]  # (n, T, num_units) or (n, num_ts, N)
    targets = gt["targets"][gt_idx]  # same shape

    if predictions.shape != targets.shape:
        raise ValueError(
            f"Shape mismatch after alignment: predictions {predictions.shape} vs GT {targets.shape}."
        )

    # flatten batch and window dims -> (samples, units). Predictions may be stored at
    # reduced precision (e.g. fp16); upcast before metrics, since exp() summed over many
    # samples can overflow fp16's range even when no individual prediction is extreme.
    pred_2d = predictions.reshape(-1, predictions.shape[-1]).float()
    target_2d = targets.reshape(-1, targets.shape[-1]).float()

    # skip expensive per-call NaN/inf scan, assuming predictions are already validated
    poisson_d2 = PoissonD2Score(validate_finite=False)
    bps = BPS(validate_finite=False)
    poisson_d2.update(pred_2d, target_2d)
    bps.update(pred_2d, target_2d)

    return {
        "poisson_d2": poisson_d2.compute().item(),
        "bps": bps.compute().item(),
    }


def score_dir(
    pred_dir: str | Path,
    gt_dir: str | Path,
) -> dict[tuple[str, str, str, int], dict[str, float]]:
    """Score all TS2 prediction files found under ``pred_dir``.

    Expects prediction files written by :class:`PredictionsWriter` (metadata carries
    ``label``, ``task``, ``recording_id``, ``seed``). ``pred_dir`` may point at any subtree
    (e.g. the shared ``predictions/`` root), files whose ``task`` isn't a ``ts2-*`` task
    are skipped.

    Ground truth files are looked up as:
    ``{gt_dir}/{task}/{recording_id}/ground_truth.safetensors``

    Returns a dict keyed by ``(label, task, recording_id, seed)``, where each value is a
    dict mapping metric name to scalar value.
    """
    pred_dir = Path(pred_dir)
    gt_dir = Path(gt_dir)
    console = Console()

    results = {}
    pred_paths = sorted(pred_dir.rglob("seed_*.safetensors"))

    if not pred_paths:
        console.print(f"[yellow]No prediction files found under {pred_dir}[/yellow]")
        return results

    for pred_path in pred_paths:
        with safe_open(str(pred_path), framework="pt") as f:
            meta = f.metadata()
        task = meta["task"]
        if not is_task_of("ts2", task):
            continue

        label = meta["label"]
        recording_id = meta["recording_id"]
        seed = int(meta["seed"])

        gt_path = gt_dir / task / recording_id / "ground_truth.safetensors"
        if not gt_path.exists():
            console.print(f"[yellow]GT not found, skipping:[/yellow] {gt_path}")
            continue

        try:
            results[(label, task, recording_id, seed)] = score_file(pred_path, gt_path)
        except Exception as e:
            console.print(f"[red]Error scoring {pred_path}:[/red] {e}")

    return results


def summarize(
    raw: dict[tuple[str, str, str, int], dict[str, float]],
) -> dict[tuple[str, str, str], MetricSummary]:
    """Aggregate per-seed scores into mean ± SEM per (label, task, recording_id).

    Returns a dict keyed by ``(label, task, recording_id)`` where each value maps
    metric name to ``(mean, sem, n)``. ``sem`` is ``None`` when ``n == 1``.
    """
    grouped: dict[tuple[str, str, str], dict[str, list[float]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for (label, task, recording_id, _seed), metrics in raw.items():
        for metric_name, value in metrics.items():
            grouped[(label, task, recording_id)][metric_name].append(value)

    result: dict[tuple[str, str, str], MetricSummary] = {}
    for (label, task, recording_id), metric_vals in grouped.items():
        result[(label, task, recording_id)] = {}
        for metric_name, values in metric_vals.items():
            n = len(values)
            mean = float(np.mean(values))
            sem = float(np.std(values, ddof=1) / np.sqrt(n)) if n > 1 else None
            result[(label, task, recording_id)][metric_name] = (mean, sem, n)

    return result


def print_results(
    summary: dict[tuple[str, str, str], MetricSummary],
    console: Console | None = None,
) -> None:
    """Print a summary table with one row per (label, task, recording_id)."""
    console = console or Console()

    all_metrics: list[str] = []
    for metric_vals in summary.values():
        for m in metric_vals:
            if m not in all_metrics:
                all_metrics.append(m)

    table = Table(title="TS2 Scoring Results", show_header=True, show_lines=False)
    table.add_column("Label", style="magenta", no_wrap=True)
    table.add_column("Task", style="cyan", no_wrap=True)
    table.add_column("Recording", style="yellow", no_wrap=True)
    for m in all_metrics:
        table.add_column(m, style="green", justify="right")
    table.add_column("n", style="dim", justify="right")

    for (label, task, recording_id), metric_vals in sorted(summary.items()):
        metric_cells = []
        n = 1
        for m in all_metrics:
            if m not in metric_vals:
                metric_cells.append("[dim]--[/dim]")
                continue
            mean, sem, n = metric_vals[m]
            sem_str = f"± {sem:.4f}" if sem is not None else "[dim]± --[/dim]"
            metric_cells.append(f"{mean:.4f} {sem_str}")
        table.add_row(label, task, recording_id, *metric_cells, str(n))

    console.print(table)


def main():
    parser = argparse.ArgumentParser(
        description="Score TS2 predictions against ground truth.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--pred_dir", type=str, required=True, help="Root predictions directory.")
    parser.add_argument("--gt_dir", type=str, required=True, help="Root ground truth directory.")
    args = parser.parse_args()

    raw = score_dir(args.pred_dir, args.gt_dir)
    if raw:
        print_results(summarize(raw))


if __name__ == "__main__":
    main()
