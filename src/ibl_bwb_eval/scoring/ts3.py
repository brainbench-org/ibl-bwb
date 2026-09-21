"""TS3 scoring utility.

Loads prediction .safetensors files, aligns by unit_id to the ground truth,
and computes brain-region classification metrics.

Usage:
    python -m ibl_bwb_eval.scoring.ts3 --pred_dir predictions/<label> --gt_dir ground_truth/
"""

import argparse
from collections import defaultdict
from pathlib import Path

import numpy as np
from rich.console import Console
from rich.table import Table
from safetensors import safe_open
from safetensors.torch import load_file

from ibl_bwb_eval.entity_ids import decode_entity_ids
from ibl_bwb_eval.scoring._types import MetricSummary
from ibl_bwb_eval.tasks import CLASSIFICATION_METRICS, get_ts3_readout_spec, is_task_of, task_of

__api_ref__ = {
    "description": None,
    "sections": [{"title": None, "autosummary": ["score_file", "score_dir", "summarize"]}],
}


# The report tables below are classification-shaped, so they read the classification names
# directly. Scoring itself is not: it calls whatever metrics the task's spec declares.
MACRO_METRICS: tuple[str, ...] = tuple(f"macro/{m}" for m in CLASSIFICATION_METRICS)


def score_file(pred_path: Path, gt_path: Path) -> dict[str, float]:
    """Score one prediction file against its ground truth file.

    Args:
        pred_path: path to a predictions .safetensors file.
        gt_path: path to the ground_truth.safetensors file.

    Returns:
        Dict mapping metric name to scalar value.
    """
    with safe_open(str(pred_path), framework="pt") as f:
        pred_meta = f.metadata()
    with safe_open(str(gt_path), framework="pt") as f:
        gt_meta = f.metadata()

    if pred_meta["label_names"] != gt_meta["label_names"]:
        raise ValueError(
            "label_names mismatch between prediction and ground truth: "
            "ensure both were produced with the same canonical label ordering.\n"
            f"  pred: {pred_meta['label_names']}\n"
            f"  gt:   {gt_meta['label_names']}"
        )

    pred = load_file(str(pred_path))
    gt = load_file(str(gt_path))

    pred_uids = decode_entity_ids(pred["entity_ids"])
    gt_uids = decode_entity_ids(gt["entity_ids"])

    uid_to_pred_idx = {uid: i for i, uid in enumerate(pred_uids)}
    try:
        pred_order = np.array([uid_to_pred_idx[uid] for uid in gt_uids])
    except KeyError as e:
        raise ValueError(f"GT unit {e} not found in predictions") from e

    label_names = gt_meta["label_names"].split(",")
    pred_proba = pred["pred_proba"][pred_order].numpy()
    targets = gt["targets"].numpy()
    target_labels = [label_names[i] for i in targets]
    pred_labels = [label_names[i] for i in np.argmax(pred_proba, axis=1)]

    spec = get_ts3_readout_spec(task_of("ts3", pred_meta["task"]))
    return spec.score(target_labels, pred_labels)


def score_dir(
    pred_dir: str | Path,
    gt_dir: str | Path,
) -> dict[tuple[str, int], dict[str, float]]:
    """Score all prediction files found under ``pred_dir``.

    Expects prediction files written by :class:`PredictionsWriter` (metadata carries
    ``label``, ``task``, ``seed``). ``pred_dir`` may point at any subtree.

    Ground truth files are looked up as: ``{gt_dir}/{task}/ground_truth.safetensors``.
    Prediction files whose ``task`` doesn't start with ``ts3-`` (e.g. other task suites
    picked up under a shared ``pred_dir``) are skipped silently. Among ``ts3-`` tasks,
    one that doesn't resolve to an existing ground truth file is skipped with a warning.

    Returns a dict keyed by ``(label, seed)`` mapping metric name to scalar value.
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
        label = meta["label"]
        task = meta["task"]
        seed = int(meta["seed"])

        if not is_task_of("ts3", task):
            continue

        gt_path = gt_dir / task / "ground_truth.safetensors"
        if not gt_path.exists():
            console.print(f"[yellow]GT not found, skipping:[/yellow] {gt_path}")
            continue

        try:
            results[(label, seed)] = score_file(pred_path, gt_path)
        except Exception as e:
            console.print(f"[red]Error scoring {pred_path}:[/red] {e}")

    return results


def summarize(
    raw: dict[tuple[str, int], dict[str, float]],
) -> dict[str, MetricSummary]:
    """Aggregate per-seed scores into mean ± SEM per label.

    Returns a dict keyed by ``label`` where each value maps metric name to
    ``(mean, sem, n)``. ``sem`` is ``None`` when ``n == 1``.
    """
    grouped: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    for (label, _seed), metrics in raw.items():
        for metric_name, value in metrics.items():
            grouped[label][metric_name].append(value)

    result: dict[str, MetricSummary] = {}
    for label, metric_vals in grouped.items():
        result[label] = {}
        for metric_name, values in metric_vals.items():
            n = len(values)
            mean = float(np.mean(values))
            sem = float(np.std(values, ddof=1) / np.sqrt(n)) if n > 1 else None
            result[label][metric_name] = (mean, sem, n)

    return result


def _fmt(mean: float, sem: float | None) -> str:
    sem_str = f"± {sem:.4f}" if sem is not None else "[dim]± --[/dim]"
    return f"{mean:.4f} {sem_str}"


def print_results(
    summary: dict[str, MetricSummary],
    console: Console | None = None,
) -> None:
    """Print results: one macro summary table, one per-class table (regions as rows)."""
    console = console or Console()

    # Collect region names from per-class keys (e.g. "VISp/precision" -> "VISp")
    all_regions: list[str] = []
    for metric_vals in summary.values():
        for m in metric_vals:
            if "/" in m and not m.startswith("macro/"):
                region = m.split("/")[0]
                if region not in all_regions:
                    all_regions.append(region)

    # --- Macro summary table (one row per label) ---
    macro_table = Table(title="TS3 Macro Summary", show_header=True, show_lines=False)
    macro_table.add_column("Label", style="magenta", no_wrap=True)
    for m in MACRO_METRICS:
        macro_table.add_column(m, style="green", justify="right")
    macro_table.add_column("n", style="dim", justify="right")

    for label, metric_vals in sorted(summary.items()):
        cells = []
        n = 1
        for m in MACRO_METRICS:
            if m not in metric_vals:
                cells.append("[dim]--[/dim]")
                continue
            mean, sem, n = metric_vals[m]
            cells.append(_fmt(mean, sem))
        macro_table.add_row(label, *cells, str(n))

    console.print(macro_table)

    # --- Per-class table (one row per region, columns per label) ---
    for label, metric_vals in sorted(summary.items()):
        n = next(v[2] for v in metric_vals.values())
        table = Table(title=f"TS3 Per-Class: {label} (n={n})", show_header=True, show_lines=False)
        table.add_column("Region", style="cyan", no_wrap=True)
        table.add_column("precision", style="green", justify="right")
        table.add_column("recall", style="green", justify="right")
        table.add_column("f1-score", style="green", justify="right")

        for region in sorted(all_regions):
            cells = []
            for metric in CLASSIFICATION_METRICS:
                key = f"{region}/{metric}"
                if key not in metric_vals:
                    cells.append("[dim]--[/dim]")
                    continue
                mean, sem, _ = metric_vals[key]
                cells.append(_fmt(mean, sem))
            table.add_row(region, *cells)

        console.print(table)


def main():
    parser = argparse.ArgumentParser(
        description="Score TS3 predictions against ground truth.",
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
