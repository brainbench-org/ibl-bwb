"""TS1 scoring utility.

Loads prediction .safetensors files, matches them to pre-generated ground truth
files by trial_id, and computes all metrics defined in the TS1ReadoutSpec.

Usage:
    python -m ibl_bwb_eval.scoring.ts1 --pred_dir predictions/ts1 --gt_dir ground_truth/ts1
    python -m ibl_bwb_eval.scoring.ts1 --pred_dir predictions/ts1/mlp-baseline/ts1-licking_rate --gt_dir ground_truth/ts1
"""

import argparse
from collections import defaultdict
from pathlib import Path

import numpy as np
from rich.console import Console
from rich.table import Table
from safetensors import safe_open
from safetensors.torch import load_file

from ibl_bwb_eval.metrics import aggregate_metrics
from ibl_bwb_eval.scoring._types import MetricSummary
from ibl_bwb_eval.tasks import DataType, TargetResolution, get_ts1_readout_spec, is_task_of

__api_ref__ = {
    "description": None,
    "sections": [{"title": None, "autosummary": ["score_file", "score_dir", "summarize"]}],
}


def score_file(task: str, pred_path: str | Path, gt_path: str | Path) -> dict[str, float]:
    """Score one prediction file against its ground truth file.

    Args:
        task: Flattened task id (e.g. ``"ts1-choice"``).
        pred_path: Path to a ``seed_N.safetensors`` prediction file.
        gt_path: Path to the matching ``ground_truth.safetensors`` file.

    Returns:
        Dict mapping metric name to scalar value.

    Raises:
        ValueError: If the task declares a mask the ground truth does not carry, or
            if the mask leaves no timestep to score.
    """
    pred = load_file(str(pred_path))
    gt = load_file(str(gt_path))
    spec = get_ts1_readout_spec(task.split("-", 1)[1])

    # align predictions and ground truth by trial_id
    _, pred_idx, gt_idx = np.intersect1d(
        pred["trial_id"].numpy(), gt["trial_id"].numpy(), return_indices=True
    )

    predictions = pred["predictions"][pred_idx]  # (N, 1, D) or (N, T, D)
    values = gt["values"][gt_idx]  # (N, 1)    or (N, T, D)

    if spec.mask_key is not None:
        if "mask" not in gt:
            raise ValueError(
                f"{task} declares mask_key={spec.mask_key!r} but {gt_path} carries no "
                "'mask'. Scoring it unmasked would silently include the low-confidence "
                "timesteps the mask exists to drop."
            )
        mask = gt["mask"][gt_idx]
    else:
        mask = None

    metrics = {name: ctor() for name, ctor in spec.metrics.items()}

    if spec.target_resolution == TargetResolution.TIMESTEP:
        N, T, D = predictions.shape
        predictions = predictions.reshape(N * T, D)
        values = values.reshape(N * T, D)
        if mask is not None:
            mask = mask.reshape(-1)
            if not mask.any():
                raise ValueError(f"{task} in {gt_path} has no unmasked timestep to score.")
    else:
        predictions = predictions.squeeze(1)  # (N, D) logits for classification

    for metric in metrics.values():
        if mask is not None:
            metric.update(predictions[mask], values[mask])
        elif spec.data_type in (DataType.BINARY, DataType.MULTINOMIAL):
            metric.update(predictions, values.squeeze(-1).long())
        else:
            metric.update(predictions, values)

    return aggregate_metrics(metrics)


def score_dir(
    pred_dir: str | Path,
    gt_dir: str | Path,
) -> dict[tuple[str, str, str, int], dict[str, float]]:
    """Score all TS1 prediction files found under ``pred_dir``.

    Expects prediction files written by :class:`PredictionsWriter` (metadata carries
    ``label``, ``task``, ``recording_id``, ``seed``). ``pred_dir`` may point at any subtree
    (e.g. the shared ``predictions/`` root), files whose ``task`` isn't a ``ts1-*`` task
    are skipped.

    Ground truth files are looked up as:
    ``{gt_dir}/{task}/{recording_id}/ground_truth.safetensors``

    Returns a dict keyed by ``(label, task, recording_id, seed)``, where each value is a dict mapping metric name to scalar value.
    """
    pred_dir = Path(pred_dir)
    gt_dir = Path(gt_dir)
    console = Console()

    results = {}
    failed: list[Path] = []
    pred_paths = sorted(pred_dir.rglob("seed_*.safetensors"))

    if not pred_paths:
        console.print(f"[yellow]No prediction files found under {pred_dir}[/yellow]")
        return results

    for pred_path in pred_paths:
        with safe_open(str(pred_path), framework="pt") as f:
            meta = f.metadata()
        task = meta["task"]
        if not is_task_of("ts1", task):
            continue

        label = meta["label"]
        recording_id = meta["recording_id"]
        seed = int(meta["seed"])

        gt_path = gt_dir / task / recording_id / "ground_truth.safetensors"
        if not gt_path.exists():
            console.print(f"[yellow]GT not found, skipping:[/yellow] {gt_path}")
            continue

        try:
            results[(label, task, recording_id, seed)] = score_file(task, pred_path, gt_path)
        except Exception as e:
            failed.append(pred_path)
            console.print(f"[red]Error scoring {pred_path}:[/red] {e}")

    if failed:
        console.print(
            f"[red]{len(failed)} of {len(pred_paths)} prediction files failed to score "
            "and are absent from the results below.[/red]"
        )

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

    # collect all metric names in insertion order (union across rows)
    all_metrics: list[str] = []
    for metric_vals in summary.values():
        for m in metric_vals:
            if m not in all_metrics:
                all_metrics.append(m)

    table = Table(title="TS1 Scoring Results", show_header=True, show_lines=False)
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
        description="Score TS1 predictions against ground truth.",
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
