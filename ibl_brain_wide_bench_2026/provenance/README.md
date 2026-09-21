# How the published build was made

A record of what was run, not instructions to run it. To run the pipeline yourself, see
the [pipeline README](../README.md).

## The 0.1.0 build

Two passes over the same sessions from the root of the repository, differing only in unit
filtering and destination:

```shell
# all units, for TS1
brainsets prepare ./ibl_brain_wide_bench_2026 --local --processed-dir <data_root>/all_units

# selected units, for TS2 and TS3
brainsets prepare ./ibl_brain_wide_bench_2026 --local --processed-dir <data_root>/selected_units --unit-filter all
```

Both trees were then uploaded to `s3://brain-wide-bench/brainsets/`, under `all_units` and
`selected_units`. The LFP tier was built separately, split out of the lfpack mild-tier
archive one recording at a time, and sits alongside them under `lfp/`. See the
[dataset guide](https://brainbench-org.github.io/ibl-bwb/guides/dataset.html) for how to
download any of them.

## The split

`eid_split.py` produced `data/pretrain_eids.txt` and `data/eval_eids.txt`. It holds the
seed, the QC conditions a subject had to meet, and the three cerebellar subjects added by
hand to keep CB coverage in eval. It is kept as a record: rerunning it writes only
`splits.csv` beside itself and touches neither EID list.
