from torch_brain.data import Data

from core.dataset import IBLBrainWideBench2026
from core.model import BaseModel
from ibl_bwb_eval.tasks import ReadoutSpec


class MyModel(BaseModel):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim
        # TODO: define your architecture here

    def link_datasets(
        self,
        train_dataset: IBLBrainWideBench2026,
        val_dataset: IBLBrainWideBench2026,
        test_dataset: IBLBrainWideBench2026 | None = None,
    ):
        # TODO: initialize any vocab or dataset-dependent state here
        pass

    def configure_readout(self, readout_spec: ReadoutSpec):
        # TODO: configure output projection for evaluation
        pass

    def load_ckpt(self, ckpt: dict):
        # TODO: copy the pretrained weights out of the checkpoint, for evaluation
        pass

    def input_fn(self, data: Data) -> dict:
        # TODO: convert a Data slice into model inputs (runs on CPU before collation)
        return {"model_inputs": {}}

    def forward(self, **kwargs):
        # TODO: implement forward pass
        raise NotImplementedError
