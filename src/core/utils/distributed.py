import torch
import torch.distributed as dist
from torch import Tensor

from core.utils.logger import get_cli_logger

# Taken from
# https://github.com/facebookresearch/ijepa/blob/52c1ae95d05f743e000e8f10a1f3a79b10cff048/src/utils/distributed.py


def setup_ddp(rank, world_size, ddp_cfg):
    import os

    if (world_size > 1) or ddp_cfg.force:
        os.environ["MASTER_ADDR"] = str(ddp_cfg.master_addr)
        os.environ["MASTER_PORT"] = str(ddp_cfg.master_port)
        dist.init_process_group(
            backend="nccl",
            rank=rank,
            world_size=world_size,
            device_id=rank,
        )
    else:
        get_cli_logger().info("Skipping DDP setup")


def get_open_port() -> int:
    """Find an available port on the system.

    This function creates a temporary socket, binds it to port 0 to let the OS assign
    an available port, and returns that port number. The socket is automatically closed
    after getting the port.

    Returns:
        int: An available port number that can be used for network communication.
    """
    import socket

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("", 0))
        s.listen(1)
        return s.getsockname()[1]


def cleanup_ddp():
    if dist.is_initialized():
        dist.destroy_process_group()


def all_gather_object(x: torch.Tensor, dim: int = 0, to_cpu: bool = True) -> torch.Tensor:
    if dist.is_available() and dist.is_initialized() and (dist.get_world_size() > 1):
        # object gather is for eval/logging; detach to avoid graph + huge pickles
        x_send = x.detach().contiguous()

        if to_cpu:
            x_send = x_send.cpu()

        gathered = [None for _ in range(dist.get_world_size())]
        dist.all_gather_object(gathered, x_send)
        out = torch.cat(gathered, dim=dim)
        # put back on same device as input
        out = out.to(x.device)
        return out

    return x


class AllReduce(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        if dist.is_available() and dist.is_initialized() and (dist.get_world_size() > 1):
            x = x.contiguous() / dist.get_world_size()
            dist.all_reduce(x)
        return x

    @staticmethod
    def backward(ctx, grads):  # ty:ignore[invalid-method-override]
        return grads


def all_reduce(x: Tensor) -> Tensor:
    return AllReduce.apply(x)


def is_distributed() -> bool:
    return dist.is_available() and dist.is_initialized()


def get_world_size() -> int:
    if not is_distributed():
        return 1

    return dist.get_world_size()
