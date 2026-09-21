import hydra
import torch

from core.model import BaseModel
from core.utils.util import log_param_breakdown
from pretrain.models.mtm import MtM, MtMMasker
from ts2.ts2_eval_trainer import TS2EvalTrainer

# MtM is prompted: the forward prepends the token of the mask mode that produced the input.
# TS2 corrupts inputs the same way (co_smoothing -> "neuron", forecasting -> "causal").
TASK_MASK_MODE = {"co_smoothing": "neuron", "forecasting": "causal"}


class MtMEvalTrainer(TS2EvalTrainer):
    """TS2 trainer for MtM, prompting the forward with the task's mask-mode token.

    Training corrupts the input with the configured masker and scores the loss on the
    corrupted entries; inference always prompts with ``TASK_MASK_MODE[task]``.
    """

    def _setup_tasks(self):
        super()._setup_tasks()
        self.mask_mode = TASK_MASK_MODE[self.task]
        self.logger.info(f"Prompt token: {self.mask_mode}")

    def predict(self, X, mask=None, mask_timestamps=None, mask_units=None):
        return self.model(**X["model_inputs"], mask_mode=self.mask_mode)

    def setup_model(self, ckpt: dict | None = None):
        self.model = hydra.utils.instantiate(
            self.cfg.model,
            finetune_enable=self.cfg.finetuning.enable,
        )
        if not isinstance(self.model, MtM):
            raise TypeError(
                f"{type(self).__name__} requires a MtM model, got {type(self.model).__name__}"
            )
        self.masker = hydra.utils.instantiate(self.cfg.masker)
        if not isinstance(self.masker, MtMMasker):
            raise TypeError(
                f"{type(self).__name__} requires a MtMMasker masker, got {type(self.masker).__name__}"
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
        X["model_inputs"]["spikes"], mask, mask_mode = self.masker(
            X["model_inputs"]["spikes"], X["regions"]
        )
        pred = self.model(**X["model_inputs"], mask_mode=mask_mode)
        loss = self.loss_fn(pred, y["values"])
        return loss[mask].mean() if mask.sum() != 0 else loss.sum() * 0.0
