import hydra
import numpy as np
import wandb
from torch_brain.samplers import TrialSampler
from torch_brain.utils.binning import bin_spikes

from ibl_bwb_eval.tasks import DataType
from ts1.ts1_eval_trainer import TS1EvalTrainer

from .cebra import CEBRA


class CEBRAEvalTrainer(TS1EvalTrainer):
    """Benchmark eval trainer with in-run CEBRA encoder fitting."""

    def setup(self, ckpt):
        if self.world_size != 1:
            raise ValueError("CEBRA eval-only trainer only supports single-process execution.")
        super().setup(ckpt)

    def setup_model(self, ckpt: dict | None = None):
        self.model = hydra.utils.instantiate(self.cfg.model)
        if not isinstance(self.model, CEBRA):
            raise TypeError(
                f"{type(self).__name__} requires a CEBRA model, got {type(self.model).__name__}"
            )
        self.logger.info(f"Precision: {self.precision}")
        self.logger.info(f"Model: {self.model.__class__}")
        self.fit()
        self.collect_embeddings()
        self.model.train()

    def fit(self):
        self.logger.info("Fitting CEBRA encoder from eval train split.")
        sampler = TrialSampler(
            sampling_intervals=self.train_sampler.sampler.sampling_intervals,
            shuffle=False,
        )

        spikes = []
        behavior_targets = []
        seen_session_ids = set()
        for data_index in sampler:
            sample, target = self.train_dataset[data_index]
            seen_session_ids.add(sample.session.id)
            binned = bin_spikes(sample.spikes, len(sample.units), self.model.bin_size)
            spikes.append(binned)
            if self.model.mode == "behavior":
                behavior = self._format_behavior_target(
                    target["values"],
                    num_bins=binned.shape[0],
                    data_type=self.readout_spec.data_type,
                )
                behavior_targets.append(behavior)

        if len(spikes) == 0:
            raise ValueError("No train intervals found to fit CEBRA encoder in cebra_eval.")
        if len(seen_session_ids) != 1:
            raise ValueError(
                f"CEBRA eval-only trainer expects exactly one session, got {len(seen_session_ids)}."
            )

        session = np.concatenate(spikes, axis=0)
        behavior_labels = (
            np.concatenate(behavior_targets, axis=0) if self.model.mode == "behavior" else None
        )
        self.model.fit(session, behavior_labels=behavior_labels)

        # Deferred so importing any TS1 model does not import cebra; the model defers it too.
        import cebra

        loss_ax = cebra.plot_loss(self.model.cebra_model)
        loss_fig = loss_ax.figure if hasattr(loss_ax, "figure") else None
        if loss_fig is not None:
            self.logger.log("cebra/loss_curve", wandb.Image(loss_fig))
            self.logger.push()
        self.logger.info("Fitted CEBRA encoder on eval train split.")

    def collect_embeddings(self):
        self.model.embeddings = {}
        for split, sampler, dataset in zip(
            ["train", "val", "test"],
            [self.train_sampler, self.val_sampler, self.test_sampler],
            [self.train_dataset, self.val_dataset, self.test_dataset],
            strict=True,
        ):
            self.logger.info(f"Collecting embeddings from {split} split.")
            sampler = TrialSampler(
                sampling_intervals=sampler.sampler.sampling_intervals,
                shuffle=False,
            )
            for data_index in sampler:
                sample, _ = dataset[data_index]
                binned = bin_spikes(sample.spikes, len(sample.units), self.model.bin_size)
                embeddings = self.model.cebra_model.transform(binned)
                task_aligned_trial_idx = sample.get_nested_attribute(
                    self.readout_spec.interval_key
                ).trial_index
                assert len(task_aligned_trial_idx) == 1, (
                    f"Expected 1 trial, got {len(task_aligned_trial_idx)}"
                )
                self.model.embeddings[task_aligned_trial_idx[0]] = embeddings
            self.logger.info(f"Collected {len(sampler)} embeddings from {split} split.")

    @staticmethod
    def _format_behavior_target(values, num_bins: int, data_type: DataType) -> np.ndarray:
        discrete = data_type in (DataType.BINARY, DataType.MULTINOMIAL)
        arr = np.atleast_1d(np.asarray(values))

        if arr.ndim == 1:
            arr = arr.reshape(-1, 1)
        if arr.ndim > 2:
            raise ValueError(f"Unsupported behavior target shape: {arr.shape}")

        R = arr.shape[0]
        if R not in (1, num_bins):
            raise ValueError(
                f"Behavior target first dimension ({R}) must be 1 or num_bins ({num_bins})."
            )

        if R == 1:
            arr = np.repeat(arr, repeats=num_bins, axis=0)

        if discrete:
            return arr.squeeze(-1).astype(np.int64)  # (num_bins,)
        return arr.astype(np.float32)  # (num_bins, K)
