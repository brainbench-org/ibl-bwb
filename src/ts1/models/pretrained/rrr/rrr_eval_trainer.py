from pretrain.models.rrr import RRRDecoder
from ts1.ts1_eval_trainer import TS1EvalTrainer


class RRREvalTrainer(TS1EvalTrainer):
    """TS1 eval trainer plus the ``V`` hand-off.

    The load must follow ``configure_readout``, which creates ``V`` -- hence the
    ``super()`` call first.
    """

    def link_model(self, model: RRRDecoder, ckpt: dict | None):
        super().link_model(model, ckpt)

        if self.cfg.finetuning.enable:
            model.load_ckpt(ckpt)
