import torch
import torch.distributed as dist

from parallel.distributed import rank, world_size


def _all_reduce(x: torch.Tensor) -> torch.Tensor:
    """Return the sum of `x` across TP ranks.

    In the labs this is the operation that turns rank-local partial outputs into
    the same full tensor on every rank. The clone keeps the collective from
    mutating an upstream autograd buffer in place.
    """
    if world_size() == 1:
        return x
    y = x.contiguous().clone()
    dist.all_reduce(y, op=dist.ReduceOp.SUM)
    return y


def _split_last_dim(x: torch.Tensor) -> torch.Tensor:
    """Return this rank's shard after splitting the last tensor dimension."""
    if world_size() == 1:
        return x
    chunks = torch.chunk(x, world_size(), dim=-1)
    return chunks[rank()].contiguous()


def _gather_last_dim(x: torch.Tensor) -> torch.Tensor:
    """Gather rank-local last-dimension shards and concatenate them."""
    if world_size() == 1:
        return x
    gathered = [torch.empty_like(x) for _ in range(world_size())]
    dist.all_gather(gathered, x.contiguous())
    return torch.cat(gathered, dim=-1).contiguous()


class _CopyToTensorParallelRegion(torch.autograd.Function):
    """Identity in forward, all-reduce in backward.

    Column-parallel layers consume replicated input in forward, so no
    communication is needed before the local matmul. In backward, every rank
    produced a gradient contribution for the same replicated input, so those
    contributions must be summed.
    """

    @staticmethod
    def forward(ctx, x: torch.Tensor) -> torch.Tensor:
        return x

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> torch.Tensor:
        return _all_reduce(grad_output)


class _ReduceFromTensorParallelRegion(torch.autograd.Function):
    """All-reduce in forward, identity in backward.

    Row-parallel layers produce partial outputs. Summing them reconstructs the
    full residual-stream tensor on every rank. The backward pass receives a
    gradient for that already-replicated output, so each rank can use it
    directly for its local weight shard.
    """

    @staticmethod
    def forward(ctx, x: torch.Tensor) -> torch.Tensor:
        return _all_reduce(x.clone())

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> torch.Tensor:
        return grad_output


class _ScatterToTensorParallelRegion(torch.autograd.Function):
    """Split the last dimension in forward, gather gradient shards in backward."""

    @staticmethod
    def forward(ctx, x: torch.Tensor) -> torch.Tensor:
        return _split_last_dim(x)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> torch.Tensor:
        return _gather_last_dim(grad_output)


class _GatherFromTensorParallelRegion(torch.autograd.Function):
    """Gather last-dimension shards in forward, split gradients in backward."""

    @staticmethod
    def forward(ctx, x: torch.Tensor) -> torch.Tensor:
        return _gather_last_dim(x)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> torch.Tensor:
        return _split_last_dim(grad_output)


def copy_to_tensor_parallel_region(x: torch.Tensor) -> torch.Tensor:
    """Mark a replicated tensor as entering a tensor-parallel region."""
    return _CopyToTensorParallelRegion.apply(x)


def reduce_from_tensor_parallel_region(x: torch.Tensor) -> torch.Tensor:
    """Sum rank-local partial tensors so every rank receives the full result."""
    return _ReduceFromTensorParallelRegion.apply(x)


def scatter_to_tensor_parallel_region(x: torch.Tensor) -> torch.Tensor:
    """Shard a replicated tensor across TP ranks along its last dimension."""
    return _ScatterToTensorParallelRegion.apply(x)


def gather_from_tensor_parallel_region(x: torch.Tensor) -> torch.Tensor:
    """Reconstruct a replicated tensor from last-dimension TP shards."""
    return _GatherFromTensorParallelRegion.apply(x)
