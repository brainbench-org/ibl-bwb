"""The interface every benchmark model implements."""

from abc import ABC, abstractmethod

import optuna
import torch.nn as nn
from omegaconf import DictConfig
from torch_brain.data import Data

from core.dataset import IBLBrainWideBench2026
from ibl_bwb_eval.tasks import ReadoutSpec

__api_ref__ = {
    "description": None,
    "sections": [{"title": None, "autosummary": ["BaseModel"]}],
}


class BaseModel(nn.Module, ABC):
    """Standardized model interface.

    This class defines the recommended base interface for models using this benchmark.
    All models must inherit from it and implement :meth:`input_fn` and :meth:`forward`,
    plus whichever of :meth:`link_datasets`, :meth:`configure_readout` and
    :meth:`load_ckpt` the model needs; those three default to doing nothing.
    :meth:`create_search_space` and :meth:`process_tunable_params` are read only when
    tuning with Optuna.

    Each suite guide shows a worked example: :doc:`/guides/ts1`, :doc:`/guides/ts2` and
    :doc:`/guides/pretraining`.
    """

    def link_datasets(
        self,
        train_dataset: IBLBrainWideBench2026,
        val_dataset: IBLBrainWideBench2026,
        test_dataset: IBLBrainWideBench2026 | None = None,
    ):
        """Build whatever the model sizes from the datasets.

        Called before :meth:`configure_readout`, so what is read here sizes the readout.

        Args:
            train_dataset: Training split, the one to size from.
            val_dataset: Validation split.
            test_dataset: Test split, or None when the run scores none.
        """

    def configure_readout(self, readout_spec: ReadoutSpec):
        """Size the readout head for the task being evaluated.

        Called after :meth:`link_datasets`. The spec carries ``dim`` and
        ``num_timesteps``, so a head is sized without branching on it.

        Args:
            readout_spec: See :class:`~ibl_bwb_eval.tasks.ReadoutSpec`.
        """

    @abstractmethod
    def input_fn(self, data: Data) -> dict:
        """Convert one trial's data into the model's inputs.

        Runs per item on the dataloader workers, as a dataset transform, so it returns
        tensors rather than batches.

        Args:
            data: One trial, exposing the interval key :meth:`configure_readout` set.

        Returns:
            Dict whose ``model_inputs`` entry is splatted into :meth:`forward`. Extra
            top-level keys are the model's own to read in its trainer.
        """

    def forward(self, *args, **kwargs):
        """Run the model on one collated batch.

        Takes whatever :meth:`input_fn` put under ``model_inputs``, splatted in as
        keyword arguments, and returns what the trainer's loss reads.
        """
        raise NotImplementedError(f"{type(self).__name__} does not implement forward")

    def load_ckpt(self, ckpt: dict):
        """Copy pretrained weights out of a checkpoint into this model.

        Read only the weights: the trainer restores optimizer and epoch state itself.

        Args:
            ckpt: The loaded checkpoint, as written by the pretraining run.
        """

    @classmethod
    def create_search_space(cls, trial: optuna.Trial, cfg: DictConfig):
        """Map out the model's Optuna search space.

        Call ``trial.suggest_*``; the names suggested become the keys
        :meth:`process_tunable_params` receives.

        Args:
            trial: Optuna trial to register suggestions on.
            cfg: The run config, for values the space depends on.
        """

    @classmethod
    def process_tunable_params(cls, tune_params: dict) -> dict:
        """Turn suggested hyperparameters into config overrides.

        Runs before the config is filled, so this is where a suggestion is mapped onto the
        config path it sets (``batch_size_log2`` -> ``batch_size``), a value is derived from
        another, or a default is supplied for something not being tuned.

        Args:
            tune_params: The names :meth:`create_search_space` suggested, with their values.

        Returns:
            The overrides to apply to the config. The default returns them unchanged.
        """
        return tune_params
