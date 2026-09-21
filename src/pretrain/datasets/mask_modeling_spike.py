from typing import Any

from core.dataset import IBLBrainWideBench2026, Split


class IBLBrainWideBenchMaskModelingSpikes(IBLBrainWideBench2026):
    """Binned spikes alone, for objectives that mask them and reconstruct them.

    Samples the task-aligned windows of each pretrain recording, intersected with the
    split's domain. No behavior is read and no target is returned: what to mask, and
    what to score it against, belong to the model.

    Args:
        root: Directory holding the build.
        split: Which pretrain split to read, ``train`` or ``val``.
        dirname: Name of the build directory under ``root``.
        recording_ids: Recordings to load, or None for every pretrain recording.
        transform: Applied to each item, the model's ``input_fn`` among them.
    """

    SPLITS: tuple[Split | None, ...] = ("train", "val")

    def __init__(
        self,
        root: str,
        split: Split,
        dirname: str = "ibl_brain_wide_bench_2026",
        recording_ids: str | list[str] | None = None,
        transform: Any | None = None,
    ):
        super().__init__(
            root=root,
            dirname=dirname,
            recording_ids=recording_ids,
            transform=transform,
            split=split,
            regime="pretrain",
        )

    def get_trial_intervals(self):
        sampling_intervals = {}

        for recording_id in self.recording_ids:
            recording = self.get_recording(recording_id)

            assert "task_aligned_intervals" in recording.keys(), (  # noqa: SIM118
                f"{recording_id} has no task_aligned_intervals"
            )

            intervals = recording.task_aligned_intervals.domain
            intervals = intervals & getattr(recording, f"pretrain_{self.split}_domain")

            sampling_intervals[recording_id] = intervals

        return sampling_intervals
