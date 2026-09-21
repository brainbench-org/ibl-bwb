import hydra

from pretrain.models.ndt_stitch import NDTStitchMasker
from ts2.ts2_eval_trainer import TS2EvalTrainer


class NDTEvalTrainer(TS2EvalTrainer):
    """TS2 trainer for the single-session NDT.

    Training corrupts the input with the configured masker and scores the loss on the
    corrupted entries.
    """

    def setup(self, ckpt: dict | None = None):
        super().setup(ckpt)
        self.masker = hydra.utils.instantiate(self.cfg.masker)
        if not isinstance(self.masker, NDTStitchMasker):
            raise TypeError(
                f"{type(self).__name__} requires a NDTStitchMasker masker, got {type(self.masker).__name__}"
            )
        self.add_checkpoint_items(masker=self.masker)

    def training_step(self, X, y):
        X["model_inputs"]["spikes"], mask = self.masker(X["model_inputs"]["spikes"])
        return self.loss(self.predict(X), y["values"], mask)
