# ibl-bwb-eval

The evaluation contract for the IBL BrainWideBench benchmark: everything needed to build
a valid submission and to score one, and nothing else. It is independent of the
benchmark's training code, so it installs on a CPU-only box without torch-brain, hydra,
ray or wandb.

## Install

```bash
pip install ibl-bwb-eval --extra-index-url https://download.pytorch.org/whl/cpu
```

Without that index torch resolves from PyPI to a CUDA build, around 4 GB of NVIDIA
packages the scorer never calls, so keep the flag in automated scoring environments too.
uv needs `--index-strategy unsafe-best-match` as well, since it will not fall back to
PyPI for a package found on another index.

The import name is `ibl_bwb_eval`:

```python
import ibl_bwb_eval
```

## What it gives you

```python
import ibl_bwb_eval

ibl_bwb_eval.EVAL_RECORDING_IDS  # path to the eval session list
ibl_bwb_eval.EVAL_SEEDS          # the seeds a reported number is averaged over
ibl_bwb_eval.SELECTION_METRIC    # what model selection optimizes
ibl_bwb_eval.SUITE_TASKS         # task vocabulary, per suite
```

Reading the protocol or the task list costs no torch or numpy import. The task names and
readout specs live in `ibl_bwb_eval.tasks`, the reported metrics (bits-per-spike, Poisson D²)
in `ibl_bwb_eval.metrics`, and the write side of the submission format in
`ibl_bwb_eval.predictions.PredictionsWriter`.

## Scoring a submission

```bash
python -m ibl_bwb_eval.scoring.ts1 --pred_dir predictions/ts1 --gt_dir ground_truth/ts1
python -m ibl_bwb_eval.scoring.ts2 --pred_dir predictions/<label> --gt_dir ground_truth/ts2
python -m ibl_bwb_eval.scoring.ts3 --pred_dir predictions/<label> --gt_dir ground_truth/
# all three suites, aggregated and ranked
python -m ibl_bwb_eval.scoring.aggregation --pred_dir predictions/ --gt_dir ground_truth/
```

## Versioning

Metric definitions and readout specs are part of published numbers, so this package is
versioned as API. `COSMOS_LABELS` is the sharpest case: it is the column order of every
TS3 `pred_proba`, and the TS3 scorer rejects a submission whose `label_names` disagree
with the ground truth. Pin the version you scored with.

## License

MIT.
