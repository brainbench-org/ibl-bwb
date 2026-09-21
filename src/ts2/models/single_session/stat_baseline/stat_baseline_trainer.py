"""Trainer for the zero-parameter statistical baselines.

Everything is fit in ``link_datasets``, so the optimization path does not apply
and ``train_epoch`` is a no-op. Run with ``num_epochs=1``: the base ``train()``
loop then does one val epoch (publishing the fit's own val score) followed by
``test()``.
"""

from core.model import BaseModel
from ts2.ts2_eval_trainer import TS2EvalTrainer


class StatBaselineTrainer(TS2EvalTrainer):
    """TS2 trainer for the zero-parameter statistical baselines.

    The model is fit in ``link_datasets``, so there is no optimizer, ``train_epoch`` is a
    no-op and val reports the fit's own selection score.
    """

    def setup_optimizers(self, ckpt: dict | None):
        self.optimizer = None
        self.scheduler = None
        self.logger.info("StatBaseline: no optimizer (zero-parameter model)")
        return None, None

    def train_epoch(self):
        return

    def link_model(self, model: BaseModel, ckpt: dict | None):
        super().link_model(model, ckpt)
        self.val_score = model.val_score  # set by the fit, read before any DDP wrapping

    def val_epoch(self):
        """Report the fit's own val score instead of walking the val loader.

        The fit already scored its selection on the val split with the same metric, so
        the walk would only re-measure it at a denser stride. No bps, and nothing to
        early-stop, so ``best_model`` stays unset and ``test`` scores the fitted model
        directly.
        """
        score = self.val_score
        self.best_val_metrics = {"best/val/poisson_d2": score, "best/val/avg": score}
        self.best_metrics = self.best_val_metrics | {"best/epoch": self.epoch}
        val_metrics = {"val/poisson_d2": score, "val/avg": score, "val/epoch": self.epoch}
        self.logger.update_epoch_postfix(self.get_best_pbar_metrics())
        self.logger.log_dict(val_metrics | self.best_metrics)
        self.push_logs()
        self.return_value = val_metrics | self.best_metrics
        return False
