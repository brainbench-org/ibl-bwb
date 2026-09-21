# CLAUDE.md

Guidance for agents working in **IBL BrainWideBench (BWB)**. Read the README for what the
repo contains and the docs for how to use it, as well as relevant links to the paper, docs,
leaderboard, and baseline runs on Weights & Biases. This file is for judgment calls the
README doesn't make: what an agent must never touch without asking, and where to look instead of guessing.

Note that this guide largely focuses on users of the benchmark itself. `CONTRIBUTING.md` covers both
audiences: Part 1 ("Using the benchmark") is for anyone entering a model — submission workflow, model
layout, reporting issues — and Part 2 ("Working on the repository") is for changes to this codebase
itself — setup, PR process, releasing `ibl-bwb-eval`.

See `docs/guides/overview.html` for who owns what and how a sample reaches the model — read this before
editing `core/`, a dataset class, or a trainer.

**Always start with `docs/guides/`, not memory.** This is a research benchmark with a published, versioned
scoring contract (`ibl_bwb_eval`). Guessing at a config key, a metric definition, or a directory's purpose
and being wrong here doesn't just produce bad code — it can produce a number that looks like a benchmark
score but isn't comparable to one. When the docs and your prior knowledge of similar codebases disagree,
the docs win.

## The rule that matters most: never touch evaluation on your own initiative

If a model's benchmark numbers look wrong, bad, or surprising, **the default diagnosis is the model, not
the metric.** Do not "fix" a metric, a scoring function, a split, or seed handling to make a number look
more reasonable. Silently changing how a score is computed so it looks better is a correctness bug, not
a helpful adjustment, and it's especially dangerous because it fails silently.

Concretely:

- **Never edit** `src/ibl_bwb_eval/`, an `*_eval_trainer.py` or `*_test_mixin.py`, a dataset's train/val/test
  split logic, or seed handling, unless the user's task is explicitly *about* changing one of those and they've
  confirmed it. `CONTRIBUTING.md` calls this out too: these are exactly the files that "can change a published
  result while CI stays green."
- If the task genuinely requires a different metric or protocol (e.g. a research question the benchmark
  doesn't currently measure), **keep the benchmark's original metric computation running alongside the
  new one** rather than replacing it, or propose that as the plan before writing code. A number that
  can't be traced back to the published contract isn't a BWB number anymore.
- If you find yourself reasoning "the score would look better if the metric were computed this other
  way" — stop. That's the tell. Report the low score as a modeling result and ask the user how they want
  to proceed.
- Never fabricate, estimate, or backfill leaderboard scores. Scoring runs server-side (see "Submitting
  to the leaderboard" in the README) — if you don't have a number from the server, say so instead of
  extrapolating one.

## Ask before touching, don't just proceed

Beyond eval code, a few things are important to understand before editing. Before editing any of these,
state what you're about to change and why, and get explicit confirmation:

- **Anything under `src/ibl_bwb_eval/`**, the naming registry (`COSMOS_LABELS`, `task_id()` output, the
  `brain-wide-bench` S3 bucket layout) — these are read by code and data outside this repo.
- **A model's `_target_` path or its file/class name once it has a released checkpoint** — checkpoints
  restore their model class from the stored `_target_`, so a rename silently strands every existing
  checkpoint of that model. See the naming section of `CONTRIBUTING.md`.
- **Anything in `packaging/ibl-bwb-eval/`** or a version bump there — this is a second, independently
  versioned PyPI distribution that the leaderboard pins against. Releasing is a deliberate act, not a
  side effect of another change.

If you're unsure whether something falls in this list, treat it as if it does and ask.

## Finding data fields and data quality

Don't guess at what fields exist in a recording or what a session's QC status is. The dataset-build code
and QC tables under `ibl_brain_wide_bench_2026/` (see its own README) are the source of truth for what a
`Data` object actually carries and which sessions/units pass quality control — read the build pipeline
script and the QC CSVs there before assuming a field exists or a session is usable. The runtime side
(`core/dataset.py`, `IBLBrainWideBench2026`) only reads what the build already produced; it isn't where
you'd discover a new field. For exploring the data files themselves, refer to the torch_dataset library
API (docs: https://torch-brain.readthedocs.io/en/stable/).

## Conventions worth internalizing beyond the README

- **Suites are deliberately self-contained.** `src/ts1/`, `src/ts2/`, `src/ts3/` each own their full
  stack, including duplication of logic that looks like it belongs in `core/`. Resist consolidating
  "obviously shared" code across suites into `core/` — that's an explicit design choice here, not an
  oversight. `core/` is only for machinery that is genuinely suite-agnostic (trainer loop, checkpointing,
  batching, samplers).
- **Model directories are self-contained and copyable.** New models are added by copying
  `src/pretrain/models/my_model/`, not by threading a new model through central config. Nothing outside
  the model's own directory should need editing to add one.
- **Import style is directory-scoped, not repo-wide.** Inside a model's own directory, imports are
  relative (`from .cache import ...`) because the directory is meant to be copied and renamed. Everywhere
  else — `core/`, `ibl_bwb_eval/`, suite top levels — imports are absolute. Don't "fix" one style to match
  the other.
- **A `_target_` names a package (`pretrain.models.<name>.<Class>`), never a module.** This is checked by
  `test_config_targets.py`; if you add a model, follow the pattern rather than pointing `_target_` at a
  file.
- **Naming is fixed per layer, not per taste** — see the Naming table in `CONTRIBUTING.md` (prose vs.
  PyPI/repo vs. Python package vs. class). When in doubt about how to render "BrainWideBench" or a model
  name in a given context, check that table rather than picking a convention.
- **New baseline models aren't merged into this repo.** They belong in a fork, evaluated by submitting
  predictions to the leaderboard — not as a PR against `main`. Don't propose adding a new baseline model
  into this codebase.
- Before pushing any change (contributing): `ruff check . && ruff format --check .` and `pytest src/tests -q`.
  This is cheap to run and CI will fail without it.

## Working with different kinds of users

Identify which of these the user is doing, since it changes what "correct" looks like — but don't ask
if it's inferable from the request itself.

1. **Reading the codebase / exploring data**, no training involved. Point to `docs/guides/overview.html`
   for "who owns what" before diving into `core/`, and to the specific suite's guide (TS1/TS2/TS3) for
   task-level detail. For data exploration, prefer reading through `core/dataset.py` and a suite's dataset
   subclass over grepping — the "How a sample reaches the model" pipeline in the codebase-overview guide
   explains the transform order, which is easy to misread from the code alone. For questions about the data
   itself, point to the data pipeline in `ibl_brain_wide_bench_2026/`, and explore the data files (wherever
   they are stored) using the torch_dataset library API (docs: https://torch-brain.readthedocs.io/en/stable/).
2. **Rerunning an existing baseline.** The commands are in the README / `docs/guides/pretraining.html` and
   the per-suite guides — use the documented Hydra invocation (`trainer=<model>`) rather than
   reconstructing one from reading the training script. If a run doesn't reproduce a documented number,
   that's a `CONTRIBUTING.md`-worthy issue report (full command, `recording_id`/`task`/seed, commit hash,
   traceback), not a reason to adjust the eval code.
3. **Modifying or extending an existing baseline** (new transform, different hyperparameters, a variant
   architecture). Stays inside the model's own directory and its trainer/config files; touching the eval
   trainer or the suite's shared dataset class to special-case one model is a signal to reconsider the
   approach, and falls under "ask before touching" above if it seems unavoidable.
4. **Implementing a new model end-to-end.** Copy `src/pretrain/models/my_model/`, follow the
   `<name>.py` / `<name>_<role>.py` / `configs/` layout, implement `BaseModel`'s interface
   (`forward`, `input_fn`, `link_datasets`, `configure_readout`, `load_ckpt`) rather than reinventing it,
   and rely on `model_config_discovery.py` auto-discovery instead of registering the model anywhere
   central. Confirm with the user whether the goal is a leaderboard submission (train → `save_preds` →
   upload at bwb.iblcore.org, per the README) or just local evaluation.
5. **Bringing an external pretraining pipeline and evaluating only on BWB.** This person doesn't need
   `src/pretrain/` at all — point them at a suite's eval trainer or mixin and `BaseModel`'s `load_ckpt`/`input_fn`
   as the integration surface: wrap the external checkpoint to satisfy that interface, then follow the
   same suite-level train/eval and `save_preds` path as everyone else. Don't route them through the
   pretraining code just because it's there.

## Guidelines to follow while writing code!

Behavioral guidelines to reduce common LLM coding mistakes.

**Tradeoff:** These guidelines bias toward caution over speed. For trivial tasks, use judgment.

### 1. Think Before Coding

**Don't assume. Don't hide confusion. Surface tradeoffs.**

Before implementing:
- State your assumptions explicitly. If uncertain, ask.
- If multiple interpretations exist, present them - don't pick silently.
- If a simpler approach exists, say so. Push back when warranted.
- If something is unclear, stop. Name what's confusing. Ask.

### 2. Simplicity First

**Minimum code that solves the problem. Nothing speculative.**

- No features beyond what was asked.
- No abstractions for single-use code.
- No "flexibility" or "configurability" that wasn't requested.
- No error handling for impossible scenarios.
- If you write 200 lines and it could be 50, rewrite it.

Ask yourself: "Would a senior engineer say this is overcomplicated?" If yes, simplify.

### 3. Surgical Changes

**Touch only what you must. Clean up only your own mess.**

When editing existing code:
- Don't "improve" adjacent code, comments, or formatting.
- Don't refactor things that aren't broken.
- Match existing style, even if you'd do it differently.
- If you notice unrelated dead code, mention it - don't delete it.

When your changes create orphans:
- Remove imports/variables/functions that YOUR changes made unused.
- Don't remove pre-existing dead code unless asked.

The test: Every changed line should trace directly to the user's request.

### 4. Goal-Driven Execution

**Define success criteria. Loop until verified.**

Transform tasks into verifiable goals:
- "Add validation" → "Write tests for invalid inputs, then make them pass"
- "Fix the bug" → "Write a test that reproduces it, then make it pass"
- "Refactor X" → "Ensure tests pass before and after"

For multi-step tasks, state a brief plan:
```
1. [Step] → verify: [check]
2. [Step] → verify: [check]
3. [Step] → verify: [check]
```

Strong success criteria let you loop independently. Weak criteria ("make it work") require constant clarification.

---

**These guidelines are working if:** fewer unnecessary changes in diffs, fewer rewrites due to overcomplication, and clarifying questions come before implementation rather than after mistakes.