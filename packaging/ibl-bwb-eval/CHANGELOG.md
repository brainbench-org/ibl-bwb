# Changelog

`ibl-bwb-eval` is published to PyPI and the leaderboard scores submissions against a
pinned version, so every release needs an entry here. A version number can never be
reused and its metadata is frozen once published.

The log starts at `0.0.2`. `0.0.1` predates it and is described only by what `0.0.2`
changed about it.

What forces a release is [the section on reported
numbers](https://github.com/brainbench-org/ibl-bwb/blob/main/CONTRIBUTING.md#changes-that-can-move-a-reported-number):
a change that moves one, or that alters the submission format or the public API. Those
are the changes worth writing down here too. Internal refactors that leave the contract
and the numbers alone do not need an entry.

The version lives in `src/ibl_bwb_eval/_version.py`, and `PredictionsWriter` stamps it
into every prediction file as `ibl_bwb_eval_version`.

## Unreleased

Anything landing before the next release goes here, and the release renames this section
and bumps `_version.py`.

## 0.1.0 (2026-09-21)

The same code as 0.0.2: nothing changed in the contract, the scorers or the metrics, and
a submission built against 0.0.2 scores identically here. The number marks the release
that goes with the first public dataset build, brainset `0.1.0`, so the two halves of
the benchmark carry the same version from here.

## 0.0.2 (2026-09-21)

### Changed

- **Breaking.** `TargetLayout` is now `TargetResolution`, its members `SEQUENCE_LEVEL`
  and `TIMESTEP_LEVEL` are `SEQUENCE` and `TIMESTEP`, and the field reads
  `target_resolution`. Nothing is serialized under those names, so prediction files and
  saved embeddings written by 0.0.1 still read (#582).

### Added

- Prediction files carry the version that wrote them as `ibl_bwb_eval_version`, read back
  by `version_of()`. It returns `None` for a file written by 0.0.1, which stamped nothing
  (#542).
- The scored windows are named constants in the contract, so a model can take more
  context without moving what is scored: `TARGET_WINDOW` and `BEHAVIOR_SFREQ` from
  `ibl_bwb_eval.tasks`, and `COSMOOTH_TARGET_WINDOW` and `FORECAST_BASE_WINDOW` from
  `ibl_bwb_eval.tasks.ts2` (#470).
