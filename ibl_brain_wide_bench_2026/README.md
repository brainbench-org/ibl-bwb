# IBL BrainWideBench 2026 Pipeline

## Overview

This pipeline processes electrophysiology and behavioral data from the [IBL Brain Wide Map](https://www.internationalbrainlab.com/brain-wide-map) dataset. It downloads raw session data via the ONE API, processes spikes, trials, wheel, whisker, paw, and lick signals, and saves them as `.h5` files for downstream benchmarking.

## Layout

| Path | What it is |
|---|---|
| `pipeline.py` | The entry point. `brainsets prepare` loads this file by name, so it must stay at the root. |
| `CHANGELOG.md` | What changed at each `derived_version`, the build label stamped into every `.h5`. |
| `utils.py` | Resampling, decimation and grid-regularization helpers used by the extractors. |
| `data/` | Inputs the pipeline reads: the QC table and the four EID lists. |
| `test_pipeline_output.py` | The CI integration test: builds two eval sessions and diffs them against S3. |
| `provenance/` | How the published build was made: the split script and the commands that ran. |

The pipeline splits sessions into two sets:
- **pretrain**: sessions listed in `data/pretrain_eids.txt`
- **eval**: sessions listed in `data/eval_eids.txt`

## QC

Neural and behavior QC labels are read from `data/bwm_qc.csv` (compiled from manually collected QC metrics) and merged into the session manifest. These neural QC labels were collected by Matt Whiteway, Gaelle Chapuis, and Olivier Winter.

## Output schema conventions

Four naming rules hold across every `.h5` the pipeline writes. They are load-bearing:
downstream code builds these keys with f-strings rather than listing them.

**1. Split domains are `{scope}_{split}_domain`.**

`scope` is the consumer (`pretrain`, `ts1`, `ts2`), `split` is one of `train`, `val`,
`test`. Pretrain sessions carry only the `pretrain_*` domains (train/val, causal 90/10);
eval sessions carry both `ts1_*` (causal 40/20/40) and `ts2_*` (interleaved 5 min chunks,
40/20/40).

**2. Normalization stats are `{scope}_normalize.{value_key}`.**

Same `scope` vocabulary, so `ts1_normalize.paws.left_paw_speed` holds the mean and std
computed on `ts1_train_domain` only. Stats are stored only for the pixel-valued signals
that are actually z-scored (whisker, paws); wheel speed and licking rate are never
normalized and deliberately absent.

**3. The last component of a `value_key` is the task id.**

This holds for behavior signals and for task-aligned intervals alike:

| task id | `value_key` |
|---|---|
| `whisker_motion_energy` | `whisker.whisker_motion_energy` |
| `wheel_speed` | `wheel.wheel_speed` |
| `left_paw_speed` | `paws.left_paw_speed` |
| `licking_rate` | `licks.licking_rate` |
| `choice` | `task_aligned_intervals.choice.choice` |
| `block_prior` | `task_aligned_intervals.block_prior.block_prior` |

Signal names therefore repeat their modality (`wheel.wheel_speed`, not `wheel.speed`).

## CLI Flags

| Flag | Type | Default | Description |
|------|------|---------|-------------|
| `--reprocess` | bool flag | `False` | Force reprocessing of sessions that already have an output `.h5` file. Without this flag, already-processed sessions are skipped. |
| `--no-download-first` | bool flag | downloads enabled | Skip the up-front download step and use locally cached data. Useful when data is already present on disk and you want to run or re-run processing. The manifest item (eid + ONE handle) is still returned so `process()` can read from cache; anything missing is fetched lazily by ONE during processing, with the same retry budget as the download step. |
| `--api-retries` | int | `3` | Number of attempts for each ONE API call or download before the session fails. Applies to both the up-front download step and the lazy fetches done during processing, so `--no-download-first` is still covered. |
| `--unit-filter` | repeatable str | `[]` | Which unit quality filters to apply before saving spike data. Can be passed multiple times. Available options: `all`, `probe_qc`, `firing_rate`, `unit_qc`. Pass `all` for the full set required by TS2 and TS3, or none to keep all units. |
| `--list-sessions` | str | `False` | Path to a file listing session EIDs to process (one per line). |
| `--small` | bool flag | `False` | Use small EID lists (`data/pretrain_eids_small.txt` / `data/eval_eids_small.txt`) instead of the full lists. Useful for quick end-to-end tests of the pipeline. |
| `--example-frame` | bool flag | `False` | Extract and store an example frame (frame 1000) from the left camera as a grayscale array inside a `video` object. Disabled by default as it requires streaming the video. |

### Unit filters

- **`probe_qc`**: keeps only probes with a probe QC label equal to `PROBE_QC_LABEL` (`PASS`, i.e. "good"). This filter is *only applied to eval sessions*. Not applied by default.
- **`firing_rate`**: keeps only units with a mean firing rate above `MIN_FIRING_RATE` (1.0 Hz). Not applied by default.
- **`unit_qc`**: keeps only units with a KiloSort QC label equal to `UNIT_QC_LABEL` (1.0, i.e. "good"). Not applied by default.

Filters can be combined (e.g., pass `--unit-filter firing_rate --unit-filter unit_qc`) and are applied with a logical AND. `--unit-filter all` is shorthand for the full set.

### Recorded in the output

Each `.h5` records what was applied in its `brainset` metadata, so a build is self-describing:

- **`unit_filtering`**: `all_units` (no filter), `selected_units` (the full set for that regime), or `custom` (any other combination). `probe_qc` is eval only, so `selected_units` means `firing_rate, probe_qc, unit_qc` for eval sessions and `firing_rate, unit_qc` for pretrain ones.
- **`unit_filters`**: the filters actually applied, comma separated.
- **`unit_filter_thresholds`**: the thresholds behind them.

The loaders read this label and refuse to run TS2 or TS3 eval on anything but `selected_units`.

Each `.h5` also records how it was built, written by `get_provenance()` in `pipeline.py`:
the pipeline commit, whether the tree was dirty and a hash of the diff if it was, the
pipeline arguments, the build timestamp, `ONE_REVISION_LAST_BEFORE`, and the ONE-api,
ibllib, iblatlas, numpy and scipy versions. Read those fields rather than anything checked
in here, which cannot track a rebuild. One caveat on `pipeline_commit`: the files released
as 0.1.0 were built before this repository was published, so their commit predates the
first public commit and will not resolve here.

## Usage

> [!WARNING]
> It's important to be careful about which filters are being used. By default no filters are applied, which is expected for TS1, but we must apply filters for standard evaluation on TS2 and TS3.

For the standard baselines in **Task Suite 1**, we apply no QC filtering by default:
```bash
# Standard TS1 pipeline
brainsets prepare ./ibl_brain_wide_bench_2026 --local
```

For **Task Suites 2 and 3**, we apply all three filters:
```bash
# Standard TS2 & TS3 pipeline
brainsets prepare ./ibl_brain_wide_bench_2026 --local --unit-filter all
```

### Other configurations

Refer to the table above to see all the available options part of the `ibl_brain_wide_bench_2026` brainset.

In the `brainsets` command, you can configure raw (`--raw-dir`) and processed (`--processed-dir`) directories, as well as number of cores `-c`.
You can also configure directories before running the pipeline with `brainsets config`. Refer to the [brainsets documentation](https://brainsets.readthedocs.io/en/latest/index.html) for more details/options, or look under `brainsets --help`.

Point the raw directory at a persistent cache so ONE does not re-download between runs. `test_pipeline_output.py` does not go through the `brainsets` CLI: it takes two environment variables, falling back to a temporary directory that is discarded after the run.

```bash
export IBL_RAW_DIR=~/.cache/ibl_raw         # raw ONE downloads
export S3_H5_CACHE_DIR=~/.cache/ibl_s3_ref  # the S3 reference .h5 files the test diffs against
```

Nothing loads these from a `.env`, so export them in the shell you run the test from. The [CI job](../.github/workflows/test_pipeline.yml) sets the same two.

Brainsets will create an isolated environment in `/tmp` where it will install packages requires for the pipeline. Packages are installed with `uv`. Like above, you can avoid caching packages by setting the environment variable `UV_NO_CACHE=1`.

Here are some assorted examples:
```bash
# Run with 16 cores
brainsets prepare ./ibl_brain_wide_bench_2026 --local -c 16
```
```bash
# Skip download, reprocess all sessions
brainsets prepare ./ibl_brain_wide_bench_2026 --local --no-download-first --reprocess
```
```bash
# Quick smoke test on a small subset
brainsets prepare ./ibl_brain_wide_bench_2026 --local --small
```
```bash
# Save TS1 and TS2/TS3 data simultaneously in different processed directories
brainsets prepare ./ibl_brain_wide_bench_2026 --local --processed-dir /path/to/processed_ts1
brainsets prepare ./ibl_brain_wide_bench_2026 --local --processed-dir /path/to/processed_ts2_ts3 --unit-filter all
```
