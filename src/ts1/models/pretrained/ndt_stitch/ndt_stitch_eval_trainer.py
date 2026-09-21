import hydra
from torch_brain.transforms.container import Compose

from core.utils.util import log_param_breakdown
from pretrain.models.ndt_stitch import NDTStitch
from ts1.ts1_eval_trainer import TS1EvalTrainer


class NDTStitchEvalTrainer(TS1EvalTrainer):
    def setup_model(self, ckpt: dict | None):
        self.model = hydra.utils.instantiate(
            self.cfg.model,
            finetune_enable=self.cfg.finetuning.enable,
        )
        if not isinstance(self.model, NDTStitch):
            raise TypeError(
                f"{type(self).__name__} requires a NDTStitch model, got {type(self.model).__name__}"
            )
        self.add_checkpoint_items(model=self.model)

        self.logger.info(f"Precision: {self.precision}")
        self.logger.info(f"Model: {self.model.__class__}")
        self.model.train()

    def link_model(self, model: NDTStitch, ckpt: dict | None):
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
