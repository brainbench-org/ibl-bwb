from pathlib import Path

import numpy as np
import torch

from core.dataset import BenchmarkRegime, Split
from ibl_bwb_eval.tasks import TS3Task, check_ts3_label_order, get_ts3_readout_spec
from ts3.protocol import get_md_from_dataset
from ts3.ts3_dataset import IBLBrainWideBenchTS3


class LOLCATDataset(IBLBrainWideBenchTS3):
    """Per-unit dataset using task-aligned trial windows as LOLCAT snippets."""

    def __init__(
        self,
        data_root: str,
        regime: BenchmarkRegime,
        split: Split,
        val_fraction: float,
        cache_path: Path,
        task: TS3Task = "unit_cosmos",
        label_map: dict | None = None,
    ):
        super().__init__(root=data_root, regime=regime)

        effective_rids = self._select_rids(regime, split, val_fraction)
        md = get_md_from_dataset(Path(data_root), regime, task)

        if label_map is None:
            check_ts3_label_order(sorted(md["brain_region"].unique().tolist()), task)
            label_map = {r: i for i, r in enumerate(get_ts3_readout_spec(task).label_names)}
        self.label_map = label_map

        uids, labels, depths, probe_ids = self._load_units(effective_rids, md, label_map)
        self.uids = np.array(uids)
        self.labels = np.array(labels)
        self.depths = depths
        self.probe_ids = probe_ids
        if regime == "eval":
            assert len(self.uids) == len(md), (
                f"{len(md) - len(self.uids)} of {len(md)} scored units are absent from the "
                "eval dataset, so a submission would cover fewer units than TS3 scores"
            )

        cache = torch.load(cache_path, weights_only=False)
        uid_to_hists = dict(zip(cache["uids"], cache["isi_hists"], strict=True))
        missing = [uid for uid in self.uids if uid not in uid_to_hists]
        if missing:
            raise KeyError(
                f"{len(missing)} of {len(self.uids)} units are absent from cache "
                f"{cache_path}; their recordings were skipped or failed in "
                f"update_lolcat_cache (see 'Failed <rid>' lines). First: {missing[:5]}"
            )
        self._isi_hists = [uid_to_hists[uid] for uid in self.uids]

    def _load_units(self, rids: list[str], md, label_map: dict) -> tuple:
        mask = md["session_id"].isin(set(rids)) & md["brain_region"].isin(label_map)
        filtered = md[mask]
        return (
            filtered.index.tolist(),
            [label_map[r] for r in filtered["brain_region"]],
            filtered["depths"].to_numpy(),
            filtered["probe_ids"].to_numpy(),
        )

    def _select_rids(self, regime: BenchmarkRegime, split: Split, val_fraction: float) -> list[str]:
        if regime == "eval":
            return list(self.recording_ids)

        rids_by_subject: dict[str, list[str]] = {}
        for rid in self.recording_ids:
            subj = str(self.get_recording(rid).subject.id)
            rids_by_subject.setdefault(subj, []).append(rid)

        subjects = sorted(rids_by_subject.keys())
        n_val = max(1, int(len(subjects) * val_fraction))

        chosen = subjects[-n_val:] if split == "val" else subjects[:-n_val]
        return [rid for s in chosen for rid in rids_by_subject[s]]

    def __len__(self) -> int:
        return len(self.uids)

    def __getitem__(self, idx: int) -> dict:
        item = {
            "isi_hists": self._isi_hists[idx],
            "label": int(self.labels[idx]),
            "uid": self.uids[idx],
        }
        return item

    def get_unit_ids(self) -> np.ndarray:
        return self.uids

    @property
    def n_classes(self) -> int:
        return len(self.label_map)
