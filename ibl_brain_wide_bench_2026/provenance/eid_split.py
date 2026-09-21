"""One-shot script that produced ``data/pretrain_eids.txt`` and ``data/eval_eids.txt``.

Kept for provenance: it records the seed and the hand-picked CB subjects behind the
split. Rerunning it overwrites nothing; it only writes ``splits.csv`` beside itself.
"""

import random
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
csv = pd.read_csv(HERE.parent / "data" / "bwm_qc.csv")

pass_mask = csv.qc_overall == "PASS"
pass_subj = np.unique(csv[pass_mask].subject_id.values)

candidate_subj_list = list(pass_subj)
new_list = []
for subj in candidate_subj_list:
    _csv = csv[csv.subject_id == subj]

    _csv["br_qc"] = (_csv.qc_overall == "PASS") & (_csv.qc_neural_alignment == "PASS")
    cond = (_csv.groupby("eid").br_qc.apply(lambda x: x.sum()) >= 1).all()

    if cond:
        new_list.append(subj)

candidate_subj_list = new_list
print(f"Candidate subjects: {len(candidate_subj_list)}")

# seed = int(sys.argv[1])
seed = 5
print(f"{seed=}")


random.seed(seed)
random.shuffle(candidate_subj_list)
eval_subj = candidate_subj_list[:10]

# These are the only ones having CB
cb_subjects = [
    "PL030",
    "ibl_witten_26",
    "PL017",
    "CSHL053",
    "SWC_054",
    "SWC_052",
    "NYU-65",
    "NYU-40",
]
random.seed(seed)
random.shuffle(cb_subjects)
cb_subjects = cb_subjects[:3]

eval_subj.extend(cb_subjects)
eval_subj = list(set(eval_subj))
print()
print(f"Final eval subjects: {len(eval_subj)}")
print(eval_subj)

eval_mask = np.isin(csv.subject_id, eval_subj)
pretrain_csv = csv[~eval_mask]
eval_csv = csv[eval_mask]

throw_away_mask = (eval_csv.qc_overall != "PASS") | (eval_csv.qc_neural_alignment != "PASS")
throw_csv = eval_csv[throw_away_mask]
eval_csv = eval_csv[~throw_away_mask]

print()
print("Eval eids:")
print(eval_csv.eid.unique())


print()
print(
    f"Pretrain: {pretrain_csv.eid.nunique()} eids, "
    f"{pretrain_csv.probe_id.nunique()} pids\n"
    f"Eval: {eval_csv.eid.nunique()} eids, {eval_csv.probe_id.nunique()} pids\n"
    f"Throw: {throw_csv.probe_id.nunique()} / {csv.probe_id.nunique()} probes in eval"
)

throw_eids = set(throw_csv.eid.values) - set(eval_csv.eid.values)
print()
print(f"Throw eids: {len(throw_eids)}")
print(throw_eids)

pretrain_csv["split"] = "pretrain"
eval_csv["split"] = "eval"
throw_csv["split"] = "throw"
final_csv: pd.DataFrame = pd.concat([pretrain_csv, eval_csv, throw_csv])
splits_path = HERE / "splits.csv"
final_csv.to_csv(splits_path)
print(f"Saved to {splits_path}")

overall_pass = pretrain_csv[pretrain_csv.qc_overall == "PASS"]
print(f"Good overall pretrain pids: {overall_pass.probe_id.nunique()}")
print(f"Good overall pretrain eids: {overall_pass.eid.nunique()}")

neural_pass = pretrain_csv[pretrain_csv.qc_neural == "PASS"]
print(f"Good neural pretrain pids: {neural_pass.probe_id.nunique()}")
print(f"Good neural pretrain eids: {neural_pass.eid.nunique()}")

behavior_pass = pretrain_csv[pretrain_csv.qc_behavior == "PASS"]
print(f"Good behavior pretrain pids: {behavior_pass.probe_id.nunique()}")
print(f"Good behavior pretrain eids: {behavior_pass.eid.nunique()}")
