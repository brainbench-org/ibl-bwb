"""The dataset a transductive calibration trainer samples from: one eval recording."""

from functools import reduce
from operator import or_

import numpy as np
from torch_brain.data import Data, Interval

from core.dataset import UnitQCPolicy, WholeSessionSpikeDataset
from ibl_bwb_eval.tasks import TargetResolution, TS1Task, get_ts1_readout_spec

CAUSAL_TRAIN_FRAC = 0.9


class TrialAlignedCalibrationDataset(WholeSessionSpikeDataset):
    """One eval recording, sampled over its task-aligned intervals.

    This dataset offers trial-aligned sampling (same as typical sampling intervals used in
    pretraining datasets). Behavior targets are scaled, same as in pretraining. The statistics
    come from ``ts1_normalize``, per recording.

    Args:
        root: the data build to read.
        recording_ids: the one eval recording to calibrate on.
        unit_qc: unit filtering policy, passed through.
        tasks: TS1 tasks whose targets to scale, and whose domains restrict sampling. None
            leaves targets untouched and samples the task-aligned intervals whole (relevant
            for calibrations which only operate on spikes, i.e. no behavior targets).
    """

    def __init__(
        self,
        root: str,
        recording_ids: str | list[str],
        unit_qc: UnitQCPolicy | None = None,
        tasks: list[TS1Task] | None = None,
    ):
        # set before super().__init__, which reaches dataset_transform during construction
        self.tasks = list(tasks) if tasks is not None else []
        super().__init__(root=root, regime="eval", recording_ids=recording_ids, unit_qc=unit_qc)
        assert len(self.recording_ids) == 1, (
            f"calibration is one recording at a time, got {self.recording_ids}"
        )

    @property
    def recording_id(self) -> str:
        """The single recording this dataset covers."""
        return self.recording_ids[0]

    def dataset_transform(self, data: Data) -> Data:
        """Scale each task's targets to the range the pretrained readout heads expect."""
        data = super().dataset_transform(data)

        for task in self.tasks:
            readout_spec = get_ts1_readout_spec(task)

            if not data.has_nested_attribute(readout_spec.value_key):
                continue

            target = data.get_nested_attribute(readout_spec.value_key)

            if task == "licking_rate":
                target = np.round(target / 50.0).astype(np.int32)
            elif (
                readout_spec.target_resolution == TargetResolution.TIMESTEP
                and task != "wheel_speed"
            ):
                normalization = data.get_nested_attribute(f"ts1_normalize.{readout_spec.value_key}")
                target = (target - normalization.mean) / normalization.std

            if target.dtype == np.float64:
                target = target.astype(np.float32)

            data.set_nested_attribute(readout_spec.value_key, target)

        return data

    def sampling_domain(self) -> Interval:
        """Where windows may be drawn from: the task-aligned intervals, narrowed by ``tasks``."""
        recording = self.get_recording(self.recording_id)
        assert "task_aligned_intervals" in recording.keys(), (  # noqa: SIM118
            f"{self.recording_id} has no task_aligned_intervals"
        )
        intervals = recording.task_aligned_intervals.domain

        task_domains = [
            recording.get_nested_attribute(get_ts1_readout_spec(task).domain_key)
            for task in self.tasks
            if recording.has_nested_attribute(get_ts1_readout_spec(task).domain_key)
        ]
        if task_domains:
            intervals = intervals & reduce(or_, task_domains)

        return intervals

    def causal_split(
        self, train_frac: float = CAUSAL_TRAIN_FRAC
    ) -> tuple[dict[str, Interval], dict[str, Interval]]:
        """Train and val sampling intervals, split in time rather than across sessions.

        Calibration has no held-out session to validate against, so it splits this
        recording's own timeline: the first ``train_frac`` trains, the rest validates.
        Intersecting with the sampling domain keeps a gap in recording out of both sides
        rather than handing it to whichever one it falls in.
        """
        domain = self.sampling_domain()
        t0, t1 = float(domain.start.min()), float(domain.end.max())
        cutoff = t0 + train_frac * (t1 - t0)

        return (
            {self.recording_id: domain & Interval(t0, cutoff)},
            {self.recording_id: domain & Interval(cutoff, t1)},
        )
