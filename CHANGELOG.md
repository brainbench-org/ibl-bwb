# Changelog

Not a release log. This repository is never published as a package, so `git log` is the
record of what changed and this file is not a second copy of it.

What it records is the narrower set defined by [changes that can move a reported
number](CONTRIBUTING.md#changes-that-can-move-a-reported-number): anything touching
`src/ibl_bwb_eval/`, an `*_eval_trainer.py`, the dataset splits, the metrics or seed
handling. Those are the changes that silently make an older number incomparable, and the
question this file answers is "can I still compare my run to that one".

It starts at `v0.1.0`, the first public release. No number reported before it was
published, so nothing before it is described here.

The other two version lines have their own files, on their own cadence:

- The data, keyed by `derived_version`, in
  [ibl_brain_wide_bench_2026/CHANGELOG.md](ibl_brain_wide_bench_2026/CHANGELOG.md).
- The published scorer, keyed by its PyPI version, in
  [packaging/ibl-bwb-eval/CHANGELOG.md](packaging/ibl-bwb-eval/CHANGELOG.md).

Entries are dated, newest first, and say what they invalidate.

## Unreleased (0.1.0)

Anything landing before the release goes here.
