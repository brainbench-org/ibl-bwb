# Task Suite 3: Brain region decoding

Almost everything here turns a model into one embedding per unit and probes those
embeddings for the region. The exception is LOLCAT, which is supervised on the region
labels and reports F1 directly.

## Two regimes, and why they are directories

`src/ts3/models/` is split by what a model can do with a held-out unit, because that is a
property of the model rather than a flag:

- **`inductive/`** (NuCLR, NEMO, ISI). The unit embedding is a function of the unit's own
  data: spike windows, waveform and autocorrelogram, or the ISI histogram. One frozen
  checkpoint therefore represents pretrain and eval units alike, with no adaptation on the
  eval sessions.
- **`transductive/`** (POYO+, POSSM, NDT-Stitch, MtM). The unit embedding is a free
  parameter indexed by unit identity: a row of `unit_emb`, a column of a stitcher. No row
  exists for a unit the checkpoint never saw, so the eval half comes from per-session
  calibration checkpoints, one per eval recording per seed, adapted with non-region
  supervision.
- **`supervised/`** (LOLCAT). Trains in-suite on the region labels and skips embeddings.

Moving a model between the first two is not an edit to a path. For POYO+ it would mean
inventing a way to produce a unit embedding without a row for that unit.

## Calibration

A transductive extractor cannot encode an eval recording from the pretrain checkpoint
alone: the recording's units have no row in it yet. Calibration is what creates that row,
continuing the model's own pretraining objective (POYO+/POSSM's weighted multi-task
decoding loss, NDT-Stitch/MtM's masked reconstruction loss), restricted to one eval
recording via `recording_ids`, and initialized from the pretrain checkpoint by extending
its vocabulary or adding a fresh per-session stitcher rather than reinitializing from
scratch. It is a training run, not an extraction step, so it goes through `ts3/train.py`
like LOLCAT does, one call per eval recording per seed:

```bash
python src/ts3/train.py trainer=poyo_plus_calibrate data_root=<PATH> \
    ckpt.load_from=<PRETRAIN_CKPT> recording_ids=[<EID>]
```

Swap `poyo_plus` for `possm`, `ndt_stitch`, or `mtm` for the other three models; each ships
its calibration trainer and config alongside its extractor, in
`src/ts3/models/transductive/<model>/`. The `best.pt` this writes is exactly what the
extractor's `ckpt_dir` scan below is looking for. Nothing separately produces it, TS1's
and TS2's own finetuning runs included, since those finetune on a downstream decoding task
rather than continuing the pretraining objective itself.

## Extraction

One entry point, whichever model and regime:

```bash
python src/ts3/extract.py extractor=<name> data_root=<PATH> <extractor args>
```

`extractor=` resolves like `trainer=` does in TS1 and TS2, from the config each model ships
in `src/ts3/models/<regime>/<model>/configs/extractor/`. It encodes both regimes from one
extractor object, because the two halves of the file are only comparable if the same thing
produced them, and writes the four-key file described below to `<embs_dir>/<name>.pt`.

Reading pretrain units here is not a leak: this step tunes nothing, and fitting the probe on
those units is the protocol.

Inductive, from a pretrain checkpoint (see `src/pretrain/README.md` for the training runs):

```bash
python src/ts3/extract.py extractor=nuclr extractor.ckpt=<CKPT> data_root=<PATH>  # or extractor=nemo
```

Training-free, no checkpoint at all:

```bash
python src/ts3/extract.py extractor=isi data_root=<PATH>
```

Transductive, one shared pretrain checkpoint plus one calibration checkpoint per eval
recording (see Calibration above, run that first). The calibration runs are found by
scanning `extractor.ckpt_dir`, since every checkpoint records the seed and the recording it
was trained on. One seed per run, so the file is per seed:

```bash
python src/ts3/extract.py extractor=poyo_plus data_root=<PATH> \
    extractor.pretrain_ckpt=<CKPT> \
    extractor.seed=<SEED>
```

The scan keeps every `best.pt` under `ckpt_dir` whose config names one recording and the
same model as the pretrain checkpoint. It will not guess: two checkpoints for one recording
and seed raise an error listing them, so keep one sweep per directory, or narrow with
`extractor.filters={task: choice}`, which matches on any config field.

`extractor.csv_path=<CSV>` replaces the scan for a layout it cannot express. The file needs
only `ID`, `recording_ids` and `seed` columns, and may name a `ckpt_path` per row; otherwise
`<ckpt_dir>/<ID>/best.pt` is assumed.

Every eval recording must be covered: the probe join refuses an incomplete eval set, so a
missing calibration run surfaces at eval time rather than here.

LOLCAT trains instead of extracting:

```bash
python src/ts3/train.py trainer=lolcat data_root=<PATH>
```

Shared defaults live in `src/ts3/configs/train.yaml` and `src/ts3/configs/extract.yaml`;
everything a model sets for itself is in `src/ts3/models/<regime>/<model>/configs/`.

## Standardized Evaluation

Standardized evaluation runs from `src/ts3/eval.py`; the probes it fits live in
`src/ts3/probes/`.

Bringing your own embeddings needs no extractor: save a file with `torch.save` holding

```python
embs = {
    "train_embs": <torch.Tensor>  # N_train x D embeddings from the pretrain set
    "train_uids": <np.ndarray>    # N_train numpy string array, ordered like "train_embs"
    "eval_embs": <torch.Tensor>   # N_eval x D embeddings from the eval set
    "eval_uids": <np.ndarray>     # N_eval numpy string array, ordered like "eval_embs"
    "run_seed": <int>             # optional: the seed of the run behind the embeddings
}
torch.save(embs, emb_path)
```

You can then run evaluation using:

```bash
python src/ts3/eval.py probe=<linear|mlp> emb_path=<PATH>
```

One entry point fits every probe, so the multi-unit readout, the reports and the
submission files are the same whichever one you pick. What each probe sets for itself is
in `src/ts3/configs/probe/`; everything below is shared, in
`src/ts3/configs/eval.yaml`:

- `emb_path`: the embeddings file described above
- `data_root`: root directory of the dataset (default: `BWB_DATA_ROOT_SELECTED_UNITS`, else `BWB_DATA_ROOT`)
- `task`: what to classify, and at which atlas level (`unit_cosmos`, the only one scored today)
- `save_preds.enable`, `save_preds.label`: write a submission (see CONTRIBUTING.md)
- `seed`: names that submission, and is needed only for a file carrying no `run_seed`
- `wandb.*`: record the scores to W&B (see below)

Nothing is reseeded at eval time: `probe.seed` (the MLP's, the linear probe has none) keeps
its default across a sweep, so a spread is the pretraining seed's and a submission's seed is
always a pretraining seed. `probe=mlp` runs once per checkpoint: one invocation is a
`num_trials` search over `num_folds` folds plus a refit on the winner, and takes no fixed
hyperparameters, so a sweep re-searches per seed and carries that variance too.

Each probe is scored twice, single-unit and multi-unit. The multi-unit readout replaces
each unit's probabilities with the mean over its nearest neighbours on the same probe
(`ibl_bwb_eval.multi_unit`), so a probe returns probabilities and
never logits: pooling before the softmax is a different rule and gives different answers.

`probe=linear` is `sklearn` underneath, so CPU only and ~30 seconds. `probe=mlp` needs a
CUDA GPU and takes ~20 minutes on an RTX 5090, since it searches; `probe.num_trials` sets
that budget. Either way the embeddings must cover every unit the benchmark scores, or the
run raises.

### Logging to W&B

Set `wandb.mode=online` and the run is recorded like any other in the repo: one run per
probe eval, in `wandb.project`, with the full resolved config attached, so `emb_path`
records which embeddings were scored and by which probe.

Metrics are keyed `probe/<task>/<probe>_<single|multi>/<region|macro>/<metric>`, the same
names `python -m ibl_bwb_eval.scoring.ts3` computes from saved predictions. The MLP probe
also records its winning trial under `probe/<task>/mlp/`.
