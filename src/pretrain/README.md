# Pretraining

Most of these models are pretrained on the full multi-session dataset before
being finetuned on a single session for TS1 and TS2 evaluation. NuCLR and NEMO
instead learn unit-level embeddings that TS3 probes for brain region.

All commands below are run from the repo root. The only required argument is
`data_root`; everything else has a default in the trainer config.

## Released checkpoints

Every reported model's checkpoint is on the [Hugging Face Hub](https://huggingface.co/collections/nerdslab/ibl-bwb)
under `nerdslab/ibl-bwb-*`, so pretraining is optional. The
[pretraining guide](https://brainbench-org.github.io/ibl-bwb/guides/pretraining.html#released-checkpoints)
has the table, the download command, and the W&B run behind each checkpoint.

## Every model owns its trainer

A model directory holds its model, trainer and configs, and no model's trainer inherits
another's. To add one, copy `my_model/`, which subclasses `core.trainer.BaseTrainer`.

The duplication that follows is deliberate: a training protocol is part of a model's
reported result, so tuning one model must not move another's numbers. Shared code is only
machinery that does not know its caller (`BaseTrainer`, `core/` helpers,
`pretrain/datasets/`). When two trainers converge, promote the stateless part, not the loop.

## Running a pretrain

```bash
python src/pretrain/train.py trainer=<trainer> data_root=<PATH>
```

| `trainer=` | Model | `model=` (default first) |
|---|---|---|
| `ndt_stitch_pretrain` | Masked-autoencoder Transformer, block masking at 50% | `ndt_stitch_10M` (256-wide x 13 layers), `ndt_stitch_10M_wide` (512 x 5), `ndt_stitch_20M` (256 x 25) |
| `mtm_pretrain` | Region-aware masked autoencoder, mixed masking (neuron, causal, inter-region, intra-region) | `mtm_10M` (256 x 12), `mtm_10M_wide` (512 x 5), `mtm_20M` (256 x 25) |
| `ndt2_pretrain` | NDT with a simpler masking scheme and 32-step gradient accumulation | `ndt2_10M`, `ndt2_20M` |
| `neds_pretrain` | Joint pretraining on all eight TS1 tasks, 10% mask ratio | `neds_10M` |
| `poyo_pretrain` | Token-based model on one behavioral task, so `task=` is required | `poyo_10M`, `poyo_15M`, `poyo_20M` |
| `poyo_plus_multitask_pretrain` | POYO on all eight TS1 tasks at once, per-task loss weighting and unit dropout | `poyo_plus_10M` |
| `possm_multitask_pretrain` | Multi-task pretraining on all eight TS1 tasks, per-task loss weighting | `possm_10M` |
| `rrr_pretrain` | Reduced-rank linear decoder, one behavior per run, so `task=` is required | `rrr` |
| `nuclr_pretrain` | Contrastive pretraining on single-unit features, probed by TS3 | `nuclr_10M` |
| `nemo_pretrain` | Waveform and autocorrelogram views aligned with a CLIP loss, probed by TS3 | `nemo_10M` |

A non-default size is a `model=` override, and `task=` takes one of `choice`, `reward`,
`stimulus_contrast`, `whisker_motion_energy`, `wheel_speed`, `right_paw_speed`,
`left_paw_speed`, `licking_rate`:

```bash
python src/pretrain/train.py trainer=ndt_stitch_pretrain model=ndt_stitch_20M data_root=<PATH>
python src/pretrain/train.py trainer=poyo_pretrain task=choice data_root=<PATH>
```

Config locations: `src/pretrain/models/<model>/configs/trainer/<trainer>.yaml`

## Multi-task models

`poyo_plus_multitask_pretrain` and `possm_multitask_pretrain` take a subset of the eight
tasks:

```bash
python src/pretrain/train.py trainer=poyo_plus_multitask_pretrain data_root=<PATH> \
    "tasks=[choice,reward,whisker_motion_energy]"
```

The task embedding sizes from the TS1 task vocabulary rather than from `tasks=`, so
a subset run stays checkpoint-compatible with a full one. Changing the TS1 task set
itself reshapes that table and invalidates every earlier checkpoint.

POYO+'s `model.attn_impl` picks the attention backend, `nested` (default) or `xformers`;
both compute the same attention, so checkpoints carry over either way.

## RRR

RRR is phase 1 of a two-phase protocol: this run fits the shared temporal basis `V` on the
pretrain sessions, and `src/ts1` then transfers `V` to each eval recording, fitting only
that recording's `U`/`b` (`trainer=rrr_probe`). `best.pt` is the entire output of this
phase, so `ckpt.enable` defaults to true. `V`'s last axis is the task's output dimension,
which is why each behavior needs its own run and its own checkpoint.

The rank is the model's main regularizer, via `model.temporal_rank` (default 10):

```bash
python src/pretrain/train.py trainer=rrr_pretrain task=<TASK> model.temporal_rank=20 \
    data_root=<PATH>
```

## NuCLR and NEMO

Both read the `selected_units` build from `BWB_DATA_ROOT_SELECTED_UNITS`, not
`BWB_DATA_ROOT_PRETRAIN`, and both read pretrain sessions only and write checkpoints,
nothing else. Each monitors itself with a probe over pretrain units (`train/l1subout_br`),
which `monitor=null` turns off. NEMO's early stopping reads
`train/l1subout_br/f1_macro`, so `monitor=null` gives up the selection of `best.pt` along
with the probe, and its first run caches waveforms to `cache_path`; pre-build it with
`python -m pretrain.models.nemo.cache`.

Turning a checkpoint into unit embeddings is TS3's step, not a pretraining one:

```bash
python src/ts3/extract.py extractor=nuclr extractor.ckpt=<CKPT> data_root=<PATH>
```

Nothing under `src/pretrain/` names the eval regime: the extractor lives in
`ts3/models/inductive/<model>/` and is what chooses one. NEMO's encoding contract
(peak-normalisation, the ACG scale factor, the concat order) stays in
`pretrain/models/nemo/encoding.py`, shared with the online monitor. See
`src/ts3/README.md`, and `src/pretrain/models/nuclr/README.md` for NuCLR's deviations from
the official implementation.

## Common overrides

These apply to any pretrain run:

| Override | Default | Notes |
|---|---|---|
| `num_epochs=200` | 100 | Training budget |
| `base_lr=1e-4` | model-dependent | Learning rate |
| `batch_size=64` | model-dependent | Samples per batch |
| `seed=42` | 42 | Reproducibility seed |
| `ckpt.dir=<PATH>` | `ckpt/` | Checkpoint directory |
| `wandb.project=my-project` | `pretrain-<trainer>` | W&B project name |
| `wandb.mode=disabled` | env-dependent | Disable W&B logging |

DDP is enabled automatically when multiple GPUs are detected.
