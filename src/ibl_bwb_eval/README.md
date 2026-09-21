# `ibl_bwb_eval`: the evaluation contract

Everything needed to build a valid IBL BrainWideBench submission and to score one, and
nothing else. It is the lowest layer of the repo: it imports nothing from `core`,
`pretrain` or the task suites, and they import it.

```
ibl_bwb_eval/
├── protocol.py      evaluation protocol: eval seeds, sweep seed, selection metric, eval sessions
├── tasks/           one module per suite: its task names and what they predict
│                      __init__.py  task_id()/is_task_of() for the tsN- id
│                      types.py     TargetResolution / DataType
│                      ts1.py       ReadoutSpec: target keys, dimensionality, metrics
│                      ts2.py       task names only
│                      ts3.py       ReadoutSpec: the region vocabulary
├── multi_unit.py    TS3's multi-unit pooling rule, the second scored variant
├── metrics/         the reported metrics (bits-per-spike, Poisson D²)
├── predictions.py   PredictionsWriter: the write side of the submission format
├── entity_ids.py    unit-id encoding used inside those files
└── scoring/         ts1.py, ts2.py, ts3.py (the read side) and aggregation.py (ranking)
```

## Installing it on its own

It is published as `ibl-bwb-eval`, which is how anyone outside this repo gets it:

```bash
pip install ibl-bwb-eval --extra-index-url https://download.pytorch.org/whl/cpu
```

From a checkout, the equivalent is the `scoring` extra:

```bash
uv pip install -e ".[scoring]"
```

That extra is exactly this package's dependency closure: numpy, scipy, scikit-learn,
torch, torchmetrics, safetensors, rich. No torch_brain, hydra, ray or wandb, so it
installs on a CPU-only box in seconds. The extra deliberately conflicts with `train`
(see `[tool.uv] conflicts` in `pyproject.toml`), which is why nothing here may import
the training stack: `src/tests/test_ibl_bwb_eval_isolation.py` fails the build if it does.

## Scoring a submission

```bash
python -m ibl_bwb_eval.scoring.ts1 --pred_dir predictions/ts1 --gt_dir ground_truth/ts1
python -m ibl_bwb_eval.scoring.ts2 --pred_dir predictions/<label> --gt_dir ground_truth/ts2
python -m ibl_bwb_eval.scoring.ts3 --pred_dir predictions/<label> --gt_dir ground_truth/
# all three suites, aggregated and ranked
python -m ibl_bwb_eval.scoring.aggregation --pred_dir predictions/ --gt_dir ground_truth/
```

## Changing anything in here

The scorers and `PredictionsWriter` are two halves of one on-disk format, which is why
they ship together: `src/tests/test_scoring_roundtrip.py` writes with the real writer
and reads with the real scorer, so a change to either side breaks in the same PR.

Metric definitions and readout specs are part of published numbers. Changing one
invalidates every result scored before it, so treat this package as versioned API.
`COSMOS_LABELS` is the sharpest case: it is the column order of every TS3 `pred_proba`,
and `scoring/ts3.py` rejects a submission whose `label_names` disagree with the ground
truth, so reordering it invalidates every TS3 prediction file already written.
The same goes for `tasks/`: `task_id()` names the directory a submission is filed
under and `is_task_of()` is how each scorer claims its files, so renaming a task or
changing the separator orphans every prediction file already written. Both are pinned
by literal assertions in `src/tests/test_tasks.py`.
