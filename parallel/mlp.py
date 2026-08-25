import torch
import torch.distributed as dist
import torch.nn as nn
from torch.nn import functional as F

from model import MLP


def _rank() -> int:
    return dist.get_rank() if dist.is_initialized() else 0


def _world_size() -> int:
    return dist.get_world_size() if dist.is_initialized() else 1


def _all_reduce_sum(x: torch.Tensor) -> torch.Tensor:
    """Sum the same-shaped tensor across ranks and return the summed value."""
    if _world_size() == 1:
        return x
    y = x.contiguous().clone()
    dist.all_reduce(y, op=dist.ReduceOp.SUM)
    return y


def _slice_bounds(size: int) -> tuple[int, int]:
    """Return this rank's `[start, end)` slice for an evenly sharded dimension."""
    if size % _world_size() != 0:
        raise ValueError(f"Cannot evenly shard size {size} over {_world_size()} ranks.")
    shard = size // _world_size()
    start = _rank() * shard
    return start, start + shard


class TensorParallelMLP(nn.Module):
    """Minimal tensor-parallel nanoGPT MLP written directly in the MLP.

    This version intentionally avoids `ColumnParallelLinear` and
    `RowParallelLinear`. It keeps the tensor-parallel mechanics visible inside
    one small module:

    dense MLP:  x -> c_fc -> GELU -> c_proj
    TP MLP:     x -> local c_fc shard -> GELU -> local c_proj shard -> all-reduce
    """

    def __init__(self, config):
        super().__init__()
        hidden = 4 * config.n_embd
        local_hidden = hidden // _world_size()
        if hidden % _world_size() != 0:
            raise ValueError("4 * n_embd must divide evenly across TP ranks.")

        self.c_fc = nn.Linear(config.n_embd, local_hidden, bias=config.bias)
        self.gelu = nn.GELU()
        self.c_proj = nn.Linear(local_hidden, config.n_embd, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run the MLP while keeping the expanded hidden dimension sharded."""
        # `x` is replicated: every TP rank starts with the same [B, T, n_embd].
        local_hidden = self.c_fc(x)

        # GELU is elementwise, so it needs no communication across hidden shards.
        local_hidden = self.gelu(local_hidden)

        # Each rank applies only the c_proj columns matching its hidden shard.
        partial_out = F.linear(local_hidden, self.c_proj.weight, bias=None)

        # Distributed version of: full_out = partial_rank0 + partial_rank1 + ...
        out = _all_reduce_sum(partial_out)

        # Add bias after the all-reduce so it is counted once, not once per rank.
        if self.c_proj.bias is not None:
            out = out + self.c_proj.bias

        return self.dropout(out)


@torch.no_grad()
def copy_mlp_to_tensor_parallel(src: MLP, dst: TensorParallelMLP) -> None:
    """Copy dense nanoGPT MLP weights into this rank's minimal TP shards."""
    fc_start, fc_end = _slice_bounds(src.c_fc.out_features)
    proj_start, proj_end = _slice_bounds(src.c_proj.in_features)

    dst.c_fc.weight.copy_(src.c_fc.weight[fc_start:fc_end, :])
    if src.c_fc.bias is not None and dst.c_fc.bias is not None:
        dst.c_fc.bias.copy_(src.c_fc.bias[fc_start:fc_end])

    dst.c_proj.weight.copy_(src.c_proj.weight[:, proj_start:proj_end])
    if src.c_proj.bias is not None and dst.c_proj.bias is not None:
        dst.c_proj.bias.copy_(src.c_proj.bias)
