# Task Suite 1: Decoding of behavior and stimulus

TS1 trains one model per session. Every run takes three arguments:
- `recording_id`: session UUID (see `src/ibl_bwb_eval/data/eval_recording_ids.txt`)
- `task`: one of `choice`, `reward`, `stimulus_contrast`, `whisker_motion_energy`,
  `wheel_speed`, `right_paw_speed`, `left_paw_speed`, `licking_rate`
- `data_root`: path to the dataset root (defaults to `BWB_DATA_ROOT_ALL_UNITS`, then
  `BWB_DATA_ROOT`)

```bash
python src/ts1/train.py trainer=linear recording_id=<EID> task=choice data_root=<PATH>
```

Config options, the sweep runners and the baseline table with the W&B run behind each
number are in the [TS1 guide](https://brainbench-org.github.io/ibl-bwb/guides/ts1.html).
All commands here are run from the repo root.

## Single session

Trained from scratch on one session, no checkpoint needed.

| `trainer=` | Model |
|---|---|
| `linear` | Single linear layer on binned spikes |
| `mlp` | Multi-layer perceptron on binned spikes |
| `tcn` | Temporal convolutional network |
| `gru` | Gated recurrent unit network |
| `cebra` | CEBRA encoder fit in-run, then an MLP readout ([README](models/single_session/cebra/README.md), needs the `cebra` extra) |
| `ndt_superv` | Supervised single-session Transformer |
| `poyo` | POYO 1M, no pretraining |
| `possm` | POSSM 10M, no pretraining |

## Pretrained

Each of these adapts a checkpoint pretrained in `src/pretrain`, and differs only in which
parameters are trained and when.

| `trainer=` | Model | Strategy |
|---|---|---|
| `ndt_stitch_finetune` | `ndt_stitch_10M` | Full finetuning |
| `mtm_finetune` | `mtm_10M` | Full finetuning |
| `ndt2_finetune` | `ndt2_10M` | Full finetuning |
| `ndt2_linear_probe` | `ndt2_10M` | Readout only |
| `ndt2_decoder_probe` | `ndt2_10M` | Decoder and readout |
| `ndt2_gradual_unfreezing` | `ndt2_10M` | Unfreeze at `finetuning.strategy.unfreeze_at_epoch` |
| `neds_gradual_unfreezing` | `neds_10M` | Gradual unfreezing |
| `possm_gradual_unfreezing` | `possm_10M` | Gradual unfreezing |
| `poyo_gradual_unfreezing` | `poyo_10M` | Gradual unfreezing |
| `poyo_plus_gradual_unfreezing` | `poyo_plus_10M` | Gradual unfreezing |
| `rrr_probe` | `rrr` | Per-session `U`/`b` on a frozen shared basis `V` |

`ndt_stitch_finetune`, `mtm_finetune`, `possm_gradual_unfreezing` and
`poyo_plus_gradual_unfreezing` resolve `ckpt.load_from` under `BWB_PRETRAIN_CKPT_DIR` (set
in `.env`). Loading is all it affects: runs still save to `ckpt.dir`/`BWB_CKPT_DIR`. The
rest leave it unset, so pass a path like `<wandb_run_id>/last.pt` relative to `ckpt.dir`,
or an absolute one.

## Layout

Config locations: `src/ts1/models/<regime>/<model>/configs/trainer/<trainer>.yaml`, where
`<regime>` is `single_session` or `pretrained`. Every `trainer=` above is a model config
plus its trainer, so `trainer=eval model=linear` spells the same run with the generic eval
trainer.

| Entry point | What it does |
|---|---|
| `train.py` | One run: train, then the standardized test protocol |
| `tune.py` | The model's Optuna search space (`create_search_space`) over Ray |
| `scripts/train_seeds.py` | The same run over seeds 43-47, sequentially |
| `scripts/finetuning/finetune.py` | Sweep sessions, tasks and seeds with Ray, one job each |

`trainer=ndt2_calibrate` is not a baseline: it is an optional self-supervised pass over the
eval session, run before `ndt2_finetune` to adapt the checkpoint to that session. Its
trainer is the pretrain-side `NDT2Pretrain`, not a `TS1EvalTrainer`.
