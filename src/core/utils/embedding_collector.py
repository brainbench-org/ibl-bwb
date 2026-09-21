import numpy as np
import torch
import torch.distributed as dist

from core.utils.distributed import is_distributed
from core.utils.logger import get_cli_logger

logger = get_cli_logger()


class EmbeddingCollector:
    def __init__(
        self,
        unit_ids: np.ndarray | list[str],
        dim: int,
        device: torch.device,
    ):
        self.device = device
        self.unit_ids = np.sort(unit_ids)
        self.dim = dim
        self.reset()

    @torch.inference_mode()
    def update(self, x: torch.Tensor, unit_ids: np.ndarray) -> None:
        idx = np.searchsorted(self.unit_ids, unit_ids)
        idx = torch.tensor(idx, device=self.device, dtype=torch.long)

        self.emb_sum.index_add_(0, idx, x.detach().to(torch.float64))
        self.count.index_add_(0, idx, torch.ones_like(idx))

    @torch.inference_mode()
    def compute(self) -> tuple[torch.Tensor, np.ndarray]:
        if is_distributed():
            dist.all_reduce(self.emb_sum, op=dist.ReduceOp.SUM)
            dist.all_reduce(self.count, op=dist.ReduceOp.SUM)

        if not (self.count > 0).all():
            logger.warning("Some units have no embedding")
        unit_emb = self.emb_sum / self.count.unsqueeze(-1)

        self.reset()
        return unit_emb, self.unit_ids

    def reset(self):
        self.emb_sum = torch.zeros(
            (len(self.unit_ids), self.dim),
            requires_grad=False,
            device=self.device,
            dtype=torch.float64,
        )
        self.count = torch.zeros(
            len(self.unit_ids),
            requires_grad=False,
            device=self.device,
            dtype=torch.long,
        )
