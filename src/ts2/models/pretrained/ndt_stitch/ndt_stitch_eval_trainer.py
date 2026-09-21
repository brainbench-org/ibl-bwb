import hydra
import torch

from core.model import BaseModel
from core.utils.util import log_param_breakdown
from pretrain.models.ndt_stitch import NDTStitch, NDTStitchMasker
from ts2.ts2_eval_trainer import TS2EvalTrainer


class NDTStitchEvalTrainer(TS2EvalTrainer):
    """TS2 trainer for NDT Stitch.

    Training corrupts the input with the configured masker and scores the loss on the
    corrupted entries.
    """

    def setup_model(self, ckpt: dict | None = None):
        self.model = hydra.utils.instantiate(
            self.cfg.model,
            finetune_enable=self.cfg.finetuning.enable,
        )
        if not isinstance(self.model, NDTStitch):
            raise TypeError(
                f"{type(self).__name__} requires a NDTStitch model, got {type(self.model).__name__}"
            )
        self.masker = hydra.utils.instantiate(self.cfg.masker)
        if not isinstance(self.masker, NDTStitchMasker):
            raise TypeError(
                f"{type(self).__name__} requires a NDTStitchMasker masker, got {type(self.masker).__name__}"
            )
        self.add_checkpoint_items(model=self.model, masker=self.masker)

        self.logger.info(f"Precision: {self.precision}")
        self.logger.info(f"Model: {self.model.__class__}")
        self.logger.info(f"Masker: {self.masker.__class__}")
        self.model.train()

    def link_model(self, model: BaseModel, ckpt: dict | None):
        super().link_model(model, ckpt)
        if self.cfg.finetuning.enable:
            model.load_ckpt(ckpt)

        log_param_breakdown(self.logger, self.model, ("in_stitcher", "out_stitcher"))

    def training_step(self, X: dict, y: dict) -> torch.Tensor:
        X["model_inputs"]["spikes"], mask = self.masker(X["model_inputs"]["spikes"])

        loss = self.loss_fn(self.predict(X), y["values"])
        return loss[mask].mean() if mask.sum() != 0 else loss.sum() * 0.0
