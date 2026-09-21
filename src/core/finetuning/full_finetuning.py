from core.finetuning.base import FinetuningStrategy


class FullFinetuning(FinetuningStrategy):
    """No-op strategy that leaves all model parameters trainable."""

    def setup(self):
        self.logger.info("Starting full finetuning.")

    def update(self, epoch: int): ...
