import hydra
from torch_brain.transforms.container import Compose

from core.utils.util import get_num_params, get_num_trainable_params, human_readable
from pretrain.models.neds import NEDS, NEDSMasker
from ts1.ts1_eval_trainer import TS1EvalTrainer


class NEDSEvalTrainer(TS1EvalTrainer):
    def setup_model(self, ckpt: dict | None):
        modalities = ["spikes", self.cfg.task]
        self.model = hydra.utils.instantiate(
            self.cfg.model,
            finetune_enable=self.cfg.finetuning.enable,
        )
        if not isinstance(self.model, NEDS):
            raise TypeError(
                f"{type(self).__name__} requires a NEDS model, got {type(self.model).__name__}"
            )
        self.model.modalities = modalities

        self.add_checkpoint_items(model=self.model)

        self.logger.info(f"Precision: {self.precision}")
        self.logger.info(f"Model: {self.model.__class__}")
        self.model.train()

    def link_model(self, model: NEDS, ckpt: dict | None):
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
            self.train_loader.dataset,
            self.val_loader.dataset,
            self.test_loader.dataset,
        )
        model.configure_readout(self.readout_spec)
        if self.cfg.finetuning.enable:
            model.load_ckpt(ckpt)

        num_params, skipped_params = get_num_params(self.model)
        num_trainable_params = get_num_trainable_params(self.model, return_skipped=False)
        num_backbone_params = get_num_params(
            self.model.encoder, return_skipped=False
        ) + get_num_params(self.model.encoder_norm, return_skipped=False)

        self.logger.info(f"Number of parameters: {human_readable(num_params)} ({num_params:,})")
        self.logger.info(
            f"Number of trainable parameters: {human_readable(num_trainable_params)} ({num_trainable_params:,})"
        )
        self.logger.info(
            f"Number of backbone parameters (excl. embeddings & stitchers): {human_readable(num_backbone_params)} ({num_backbone_params:,})"
        )
        if skipped_params:
            self.logger.info(f"Skipped lazy parameters: {skipped_params}")

        self.masker = NEDSMasker(
            mask_ratio=0.0,
            modalities=model.modalities,
            mask_types=["behavior_masking"],
        )
        self.logger.info(f"Masker: {self.masker.__class__} (behavior_masking enforced)")

    def predict(self, X, target_timestamps=None, target_mask=None):
        model_inputs = X["model_inputs"]
        keep_masks = X["keep_masks"]
        modality_masks = self.masker(model_inputs["spikes"])
        preds = self.model(model_inputs, keep_masks, modality_masks)
        return preds[self.task]
