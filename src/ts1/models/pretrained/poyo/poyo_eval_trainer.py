import hydra
from torch_brain.transforms.container import Compose

from core.utils.util import get_num_params, get_num_trainable_params, human_readable
from ibl_bwb_eval.tasks import TargetResolution
from pretrain.models.poyo import POYO
from ts1.ts1_eval_trainer import TS1EvalTrainer


class POYOEvalTrainer(TS1EvalTrainer):
    def setup_model(self, ckpt: dict | None):
        self.model = hydra.utils.instantiate(
            self.cfg.model,
            finetune_enable=self.cfg.finetuning.enable,
        )
        if not isinstance(self.model, POYO):
            raise TypeError(
                f"{type(self).__name__} requires a POYO model, got {type(self.model).__name__}"
            )
        self.add_checkpoint_items(model=self.model)

        self.logger.info(f"Precision: {self.precision}")
        self.logger.info(f"Model: {self.model.__class__}")
        self.model.train()

    def predict(self, X, target_timestamps, *args):
        output = self.model(
            **X["model_inputs"],
            output_timestamps=target_timestamps,
            output_session_index=X["session_index"],
        )
        if self.readout_spec.target_resolution == TargetResolution.TIMESTEP:
            # queries are chained, so restore the batch axis of the target: (B, T, D)
            return output.view(*target_timestamps.shape, -1)
        # sequence-level: one query per sample, (B, D) -> (B, 1, D), to make it compatible
        # with TS1TestMixin and TS1EvalTrainer.loss().
        return output.unsqueeze(1)

    def link_model(self, model: POYO, ckpt: dict | None):
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
        num_embedding_params = get_num_params(
            self.model.unit_emb, return_skipped=False
        ) + get_num_params(self.model.session_emb, return_skipped=False)
        self.logger.info(f"Number of parameters: {human_readable(num_params)} ({num_params:,})")
        self.logger.info(
            f"Number of trainable parameters: {human_readable(num_trainable_params)} ({num_trainable_params:,})"
        )
        self.logger.info(
            f"Number of embedding parameters: {human_readable(num_embedding_params)} ({num_embedding_params:,})"
        )
        if skipped_params:
            self.logger.info(f"Skipped lazy parameters: {skipped_params}")
