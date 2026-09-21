# Brainset changelog

One entry per public `derived_version`, the literal in `pipeline.py` that every generated
`.h5` carries in its `brainset` metadata. The bucket is regenerated in place under the
same keys, so that field, not a filename or a download date, is what says which build
someone holds. Cache by ETag.

`0.1.0` is the first build released publicly. Everything before it was internal, was
never distributed, and is deliberately not described here.

This file records what changed between builds. What produced one specific file is a
different question, answered by the provenance block `get_provenance()` writes into it:
pipeline commit, tree cleanliness, arguments, timestamp, `ONE_REVISION_LAST_BEFORE`, and
the ONE-api, ibllib, iblatlas, numpy and scipy versions.

`origin_version` is the upstream IBL Brain Wide Map release, `0.0.1`.

## 0.1.0 (2026-09-20)

The first public build: 452 sessions from the IBL Brain Wide Map, 423 for pretraining and
29 held out for evaluation, in two variants of 904 files in total.

- `all_units` keeps every unit the spike sorter returned. TS1 reports on this build.
- `selected_units` applies the quality filters, firing rate and unit QC throughout and
  probe QC on the eval sessions. TS2 and TS3 require it and refuse to run without it.

What each file contains, and the split and normalization conventions behind it, is in
[README.md](README.md) and the dataset guide.
