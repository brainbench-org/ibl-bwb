# Contributing

This guide has two halves. [Using the benchmark](#part-1-using-the-benchmark) is for
anyone entering a model: how to build one, produce predictions and submit them, and how
to report a problem. [Working on the repository](#part-2-working-on-the-repository) is
for changes to this codebase itself.

Most people only need Part 1. You do not open a pull request to appear on the
leaderboard, and we do not merge new baselines.

---

# Part 1: Using the benchmark

## Submitting a model

Train your model however you like, produce prediction files, and upload them at
<https://bwb.iblcore.org>, where the leaderboard also lives. Scoring runs on our server,
so you never handle the evaluation labels and you do not upload scores of your own.

A submission covers at least three seeds. Get in touch before submitting if you cannot
run three, and say why.

**TS1 and TS2.** Training writes predictions when you enable `save_preds` and give the
run a label, which is the submission id:

```bash
python src/ts1/train.py trainer=<your_model> data_root=<path> \
    save_preds.enable=true save_preds.label=<submission-id>
```

Files land at `predictions/<label>/<task>/<recording_id>/seed_<seed>.safetensors`, under
`BWB_PREDICTIONS_DIR` (default `predictions/`).

**TS3** scores unit embeddings, so the entry point depends on what your model emits.
Embeddings (any `extractor=` under `ts3/models/inductive/` or `ts3/models/transductive/`)
are probed after the fact and the probe writes the submission:

```bash
python src/ts3/eval.py probe=linear emb_path=<emb_path> \
    save_preds.enable=true save_preds.label=<submission-id>
```

The seed comes from the run behind the embeddings, which `src/ts3/extract.py` records in
the file; pass `seed=<seed>` only for an embeddings file you built yourself. A model that
emits probabilities directly (LOLCAT) writes them from its own run with the same
`save_preds` block.

TS3 emits one label per scored variant, since single-unit and multi-unit predictions are
scored separately: `<label>_single` and `<label>_multi` from LOLCAT, and
`<label>_{linear,mlp}_{single,multi}` from the probes. Its files carry no recording-id
segment (`<label>/<task>/seed_<seed>.safetensors`), because its units are scored as one
pooled set across sessions.

## Building your own model

**We do not merge new baselines.** The models here are the ones the paper reports. Keep
yours in a fork and submit its predictions; open an issue if you would like us to review
it for validity.

Nothing central needs editing. Model configs are auto-discovered by
[model_config_discovery.py](src/hydra_plugins/model_config_discovery.py), so a directory
is enough and it becomes available as `trainer=<name>`.

```
src/pretrain/models/<name>/          src/<suite>/models/<single_session|pretrained>/<name>/
├── __init__.py                      the package, and the only thing a _target_ names
├── <name>.py                        the model
├── <name>_<role>.py                 its trainer: <name>_pretrain.py under pretrain/,
└── configs/                         <name>_eval_trainer.py in a suite
    ├── model/<name>.yaml
    └── trainer/<name>_<role>.yaml
```

Copy [src/pretrain/models/my_model/](src/pretrain/models/my_model/). Anything else the
model needs sits alongside (`masker.py`, `dataset.py`, `monitor.py`, `cache.py` are all
in use), and a second training regime is a second `<name>_<role>.py`, not a
subdirectory.

**A `_target_` names a package, never a module.** Write `pretrain.models.<name>.<Class>`
and export the class from `__init__.py`. A checkpoint stores the config it trained with,
so a target reaching into a module pins that filename forever and renaming the file
strands every checkpoint. Checked by
[test_config_targets.py](src/tests/test_config_targets.py).

**Inside a model directory, import relatively.** `from .cache import ...`. The directory
is meant to be copied and renamed, and relative imports survive that. Nowhere else:
`core/`, `ibl_bwb_eval/` and the suite top levels stay absolute.

## Reporting issues

File on the [GitHub Issues page](https://github.com/brainbench-org/ibl-bwb/issues). No
templates or labels to pick. Include:

- The full command, with every Hydra override.
- The `recording_id`, `task` and seed involved.
- Your commit (`git rev-parse --short HEAD`) and the torch and `torch-brain` versions.
- The full traceback.

Worth reporting even though they are not crashes: a number you cannot reproduce from a
documented command, and documentation that told you to do something that does not work.

---

# Part 2: Working on the repository

## Setup

```bash
source src/setup.sh <path/to/venv>     # creates the venv and installs [train]
pre-commit install
```

See the [setup guide](https://brainbench-org.github.io/ibl-bwb/guides/setup.html) for
the optional extras and the `.env` variables.

## Before you push

```bash
ruff check . && ruff format --check .   # pinned to 0.15.9, also run by pre-commit
pytest src/tests -q
```

Every PR runs [ruff](.github/workflows/ruff.yml), [tests](.github/workflows/tests.yml)
and [docs](.github/workflows/docs.yml);
[test_pipeline](.github/workflows/test_pipeline.yml) only on changes under
`ibl_brain_wide_bench_2026/`. The `tests` workflow checks that the scorer works without
the training stack, that the `ibl-bwb-eval` wheel builds, and that `src/tests` passes on
a CPU runner with no dataset.

## Pull requests

- Discuss a change in an issue first, especially a new feature.
- Work from a fork, never on `main`. Branch as `my-username/my-new-feature`, after
  `git checkout main && git pull upstream main`.
- Lint and test before pushing (see [Before you push](#before-you-push)).
- Say in the description what changed and why, and link the issues it addresses.
- Request a review in the Reviewers panel. If unsure who, ask @vinamarora8,
  @AlexandreAndre or @milosobral.

Title the PR as a
[conventional commit](https://www.conventionalcommits.org/en/v1.0.0/). Branches are
squashed, so the title is the message that lands on `main`:

```text
feat(ts3): pool probe probabilities instead of logits
fix(ndt2): skip the gradient sync between accumulation steps
docs: split the contributing guide by audience
refactor(core): restore optimizer state from the checkpoint registry
test(resume): cover a checkpoint written before the optimizer was saved
feat(ts2)!: add a 20s buffer between chunked splits
```

The scope is the suite, model or subsystem the change sits in, and `!` marks one that
invalidates existing checkpoints, predictions or published scores.

Contributions are made under the repository's [MIT license](LICENSE). No CLA or DCO is
required.

## Changes that can move a reported number

Anything touching `src/ibl_bwb_eval/`, an `*_eval_trainer.py`, the dataset splits, the
metrics, or seed handling can change a published result while CI stays green. **Open an
issue before you write the code.** The same goes for extending the benchmark, a new task
such as predicting a different brain region atlas label, for instance. Both change what
a reported number means.

Once it lands, give it an entry in [CHANGELOG.md](CHANGELOG.md), which records nothing
else: it is what tells someone whether their older number is still comparable. A change
to the data pipeline goes in
[ibl_brain_wide_bench_2026/CHANGELOG.md](ibl_brain_wide_bench_2026/CHANGELOG.md) against
the `derived_version` it ships in, and one to the published scorer in
[packaging/ibl-bwb-eval/CHANGELOG.md](packaging/ibl-bwb-eval/CHANGELOG.md).

## Task suites are meant to be self-contained

Everything governing TS2 should be in `src/ts2/`, so duplication across `ts1/`, `ts2/`
and `ts3/` is accepted on purpose. Do not move parallel code into `core/` because it
looks similar: `core/` is for the genuinely suite-agnostic machinery (trainer loop,
checkpointing, batching, samplers). The cost is that a fix in one suite's eval trainer
usually needs applying by hand to the others, so say so in the PR description.

## Naming

One name, "IBL BrainWideBench", rendered per layer rather than to taste:

| Layer | Rendering | Example |
| --- | --- | --- |
| Prose, titles, the paper | `IBL BrainWideBench` | the docs body |
| PyPI, repo, S3 buckets, URLs | hyphens | `brain-wide-bench` |
| Python packages and modules | underscores | `ibl_brain_wide_bench_2026` |
| Python classes | CamelCase, initialisms stay capital | `IBLBrainWideBenchTS1` |

- **Names that cross the repo boundary carry the `ibl` prefix**, as the repo, the
  brainset, the dataset classes and `ibl-bwb-eval` do. Internal names need not.
- **A distribution name and its import name differ only in `-` versus `_`.**
- **An initialism is total within its tier and defined here.** `IBL` everywhere, never
  spelled out. `BWB` only where the name is typed by hand (`ibl-bwb`, `ibl-bwb-eval`,
  `ibl_bwb_eval`); data identifiers, class names and prose spell it out.
- **A model has one rendering per layer, taken from the paper.** Paper casing in prose
  and class names (`NEMO`, `POSSM`, `MtM`, `NuCLR`, `NDT2`, `POYOPlus`), snake_case
  wherever it is an argument or written to disk (`model=nemo`, `trainer=possm_pretrain`).
  Wrappers keep the model's rendering and add their role: `Pretrain`, `EvalTrainer`,
  `Extractor`. Checked by [test_naming.py](src/tests/test_naming.py).

A pretrain model class name outlives a rename: TS3 rebuilds a checkpoint's model from the
`_target_` stored inside it, and NuCLR's trainer restores `cfg.model` the same way, so
renaming one invalidates every existing checkpoint of that model.

Fixed vocabulary, because something outside the repo already refers to it: the
`brain-wide-bench` bucket and its `brainsets/<build>/ibl_brain_wide_bench_2026/...` keys,
cached by ETag; the `tsN-<task>` directory names from `task_id()`, which name every
prediction file already written; and the `COSMOS_LABELS` order. `ts1`, `ts2` and `ts3`
are the only spelling of Task Suite N.

## Code style

`ruff` with the config in [pyproject.toml](pyproject.toml) is the whole style guide
(line length 100, isort, pyupgrade, bugbear, simplify). Beyond that: clarity over
cleverness, type hints on new signatures, docstrings on public API, and a regression
test with every bug fix. Comments explain *why*, not *what*, and their density stays
proportional to the surrounding code.

## Documentation

Sphinx, under [docs/](docs/):

```bash
uv pip install -e ".[docs]"
cd docs && make html        # or make html-live, which rebuilds on save
```

The API reference is generated from `__api_ref__` blocks; see
[api_reference.py](docs/source/api_reference.py) for the format.

One bib entry per method in [refs.bib](docs/source/refs.bib), keyed by the model
directory name, cited from the first line of the model class docstring
(`:cite:`ndt2``). A reference implementation is a link in the paragraph below, not a bib
entry. Trainers, extractors and eval wrappers point at the model class rather than
repeating the citation; markdown READMEs link the paper URL directly, since roles do not
render on GitHub. Generic baselines (`Linear`, `MLP`, `GRU`, `TCN`, the statistical
baselines) carry no citation. [test_citations.py](src/tests/test_citations.py) holds the
registry and fails on a key cited but undefined, or defined but never cited.

## Releasing `ibl-bwb-eval`

`src/ibl_bwb_eval/` is published to PyPI as a second distribution built from this same
checkout, with its metadata in [packaging/ibl-bwb-eval/](packaging/ibl-bwb-eval/). The
repo distribution itself is never published.

1. Bump `__version__` in `src/ibl_bwb_eval/_version.py`, the one place it is written:
   the distribution reads it as a dynamic version and `PredictionsWriter` stamps it into
   every prediction file.
2. Rename the `Unreleased` section of
   [packaging/ibl-bwb-eval/CHANGELOG.md](packaging/ibl-bwb-eval/CHANGELOG.md) to the new
   version and date it. `test_changelogs.py` fails until it exists.
3. `./packaging/ibl-bwb-eval/build.sh && uvx twine check dist/*`, both of which already
   run on every PR.
4. `uv publish`, then tag the commit as `v<version>`.

A version number can never be reused, the metadata is frozen per version, and yanking
hides a release without undoing it. What forces a release is
[the section on reported numbers](#changes-that-can-move-a-reported-number): a change
that moves one, or one that alters the submission format or the public API,
has to be reachable by version, because the leaderboard scores against a pinned one.
Step 4 is manual today and becomes Trusted Publishing once the repository move lands.
