from typing import get_args

import hydra
from torch_brain.transforms.container import Compose

from core.utils.util import log_param_breakdown
from pretrain.models.mtm import MtM, MtMMaskType
from ts1.ts1_eval_trainer import TS1EvalTrainer


class MtMEvalTrainer(TS1EvalTrainer):
    """TS1 trainer for MtM, prompting the forward with an optional mask-mode token.

    Pretraining always prompts with the mode that produced the input, but TS1 decodes
    uncorrupted input, so the mode is a config choice: ``cfg.mask_mode`` names one, or
    ``null`` leaves the sequence unprompted. See prds/ts1-mtm-trainer.md.
    """

    def setup_model(self, ckpt: dict | None):
        self.model = hydra.utils.instantiate(
            self.cfg.model,
            finetune_enable=self.cfg.finetuning.enable,
        )
        if not isinstance(self.model, MtM):
            raise TypeError(
                f"{type(self).__name__} requires a MtM model, got {type(self.model).__name__}"
            )
        self.mask_mode = self.cfg.get("mask_mode", None)
        if self.mask_mode is not None and self.mask_mode not in get_args(MtMMaskType):
            raise ValueError(
                f"mask_mode must be null or one of {list(get_args(MtMMaskType))}, "
                f"got {self.mask_mode!r}"
            )
        self.add_checkpoint_items(model=self.model)

        self.logger.info(f"Precision: {self.precision}")
        self.logger.info(f"Model: {self.model.__class__}")
        self.logger.info(f"Prompt token: {self.mask_mode}")
        self.model.train()

    def predict(self, X, target_timestamps=None, target_mask=None):
        return self.model(**X["model_inputs"], mask_mode=self.mask_mode)

    def link_model(self, model: MtM, ckpt: dict | None):
        # attach model transforms to end of dataset transform pipelines
        for split in ["train", "val", "test"]:
            model_transforms = hydra.utils.instantiate(self.cfg.get(f"{split}_transforms", []))
            self.logger.info(f"Model transforms ({split}): {model_transforms}")
            dataset = getattr(self, f"{split}_loader").dataset

            if dataset.transform is None:
                dataset.transform = Compose([*model_transforms, model.input_fn])
            else:
                dataset.transform = Compose([dataset.transform, *model_transforms, model.input_fn])

        # link datasets to model
        model.link_datasets(
            self.train_loader.dataset, self.val_loader.dataset, self.test_loader.dataset
        )
        model.configure_readout(self.readout_spec)
        if self.cfg.finetuning.enable:
            model.load_ckpt(ckpt)

        log_param_breakdown(self.logger, self.model, ("in_stitcher", "out_stitcher"))
