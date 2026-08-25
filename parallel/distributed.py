import os
from typing import Any

import torch
import torch.distributed as dist


def init_distributed(backend: str | None = None, expected_world_size: int | None = None) -> tuple[int, int]:
    """Initialize torch.distributed for small local labs.

    The repo examples default to CPU-friendly gloo so the concepts are runnable on
    a laptop. Later training examples can opt into nccl when tensors live on CUDA.
    """
    if backend is None:
        backend = "gloo"
    if not dist.is_available():
        raise RuntimeError("torch.distributed is not available in this PyTorch build.")
    if not dist.is_initialized():
        dist.init_process_group(backend=backend)

    rank = dist.get_rank()
    world_size = dist.get_world_size()
    if expected_world_size is not None and world_size != expected_world_size:
        raise RuntimeError(f"Expected {expected_world_size} ranks, got {world_size}.")
    return rank, world_size


def destroy_distributed() -> None:
    if dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()


def rank() -> int:
    return dist.get_rank() if dist.is_initialized() else 0


def world_size() -> int:
    return dist.get_world_size() if dist.is_initialized() else 1


def local_rank() -> int:
    return int(os.environ.get("LOCAL_RANK", "0"))


def fmt(value: Any) -> str:
    if torch.is_tensor(value):
        return f"shape={tuple(value.shape)} values={value.detach().cpu().tolist()}"
    return repr(value)


def ordered_print(title: str, **items: Any) -> None:
    """Print rank by rank so communication traces remain readable."""
    if not dist.is_initialized():
        rendered = ", ".join(f"{key}={fmt(value)}" for key, value in items.items())
        print(f"[rank 0] {title}: {rendered}", flush=True)
        return

    for current_rank in range(dist.get_world_size()):
        dist.barrier()
        if dist.get_rank() == current_rank:
            rendered = ", ".join(f"{key}={fmt(value)}" for key, value in items.items())
            print(f"[rank {current_rank}] {title}: {rendered}", flush=True)
        dist.barrier()


def rank0_print(text: str) -> None:
    if rank() == 0:
        print(text, flush=True)
    if dist.is_initialized():
        dist.barrier()
