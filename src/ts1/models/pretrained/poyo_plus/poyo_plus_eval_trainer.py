import hydra
from torch_brain.transforms import Compose

from core.utils.util import get_num_params, get_num_trainable_params, human_readable
from pretrain.models.poyo_plus import POYOPlus
from ts1.ts1_eval_trainer import TS1EvalTrainer


class POYOPlusEvalTrainer(TS1EvalTrainer):
    def predict(self, X, target_timestamps, *args):
        if "output_timestamps" in X["model_inputs"]:
            del X["model_inputs"]["output_timestamps"]
        if "output_session_index" in X["model_inputs"]:
            del X["model_inputs"]["output_session_index"]
        # return_dict=False + unflatten_output=True (the model's default) routes through
        # POYOPlus's unflatten logic, which yields (B, N, D) for all predictions
        # (e.g., N=1 for single-task sequence-level readouts).
        return self.model(
            **X["model_inputs"],
            output_timestamps=target_timestamps,
            output_session_index=X["session_index"],
            return_dict=False,
        )

    def link_model(self, model: POYOPlus, ckpt: dict | None):
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
            finetune_enable=self.cfg.finetuning.enable,
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
