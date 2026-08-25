import math

import torch
import torch.nn as nn
from torch.nn import functional as F

from parallel.distributed import rank, world_size
from parallel.mappings import (
    copy_to_tensor_parallel_region,
    gather_from_tensor_parallel_region,
    reduce_from_tensor_parallel_region,
    scatter_to_tensor_parallel_region,
)


def _slice_bounds(size: int, shard_rank: int, shard_world_size: int) -> tuple[int, int]:
    """Return `[start, end)` bounds for an even rank-local shard."""
    if size % shard_world_size != 0:
        raise ValueError(f"Cannot evenly shard size {size} over {shard_world_size} ranks.")
    shard = size // shard_world_size
    start = shard_rank * shard
    return start, start + shard


class ColumnParallelLinear(nn.Module):
    """Linear layer with output features sharded across tensor-parallel ranks.

    PyTorch stores nn.Linear weights as [out_features, in_features]. Column
    parallelism in the Megatron sense splits the output-feature dimension, so each
    rank owns a horizontal slice of the matrix in PyTorch layout.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        gather_output: bool = False,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.gather_output = gather_output
        self.tp_rank = rank()
        self.tp_world_size = world_size()
        self.local_out_features = out_features // self.tp_world_size
        if out_features % self.tp_world_size != 0:
            raise ValueError("out_features must divide evenly across tensor-parallel ranks.")

        self.weight = nn.Parameter(torch.empty(self.local_out_features, in_features))
        self.bias = nn.Parameter(torch.empty(self.local_out_features)) if bias else None
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in = self.in_features
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply this rank's output-feature shard of a dense linear layer.

        Input shape is unchanged and replicated across ranks, for example
        `[B, T, n_embd]`. Each rank owns `out_features / tp_world_size` output
        rows, so the local output is `[B, T, out_features / tp_world_size]`.
        Gathering is optional; the MLP deliberately leaves this hidden activation
        sharded so the following row-parallel projection can consume it directly.
        """
        x_parallel = copy_to_tensor_parallel_region(x)
        local_y = F.linear(x_parallel, self.weight, self.bias)
        if not self.gather_output or self.tp_world_size == 1:
            return local_y
        return gather_from_tensor_parallel_region(local_y)


class RowParallelLinear(nn.Module):
    """Linear layer with input features sharded across tensor-parallel ranks.

    Each rank receives the slice of x that corresponds to its local weight columns,
    computes a partial output, then all-reduces partial outputs. Bias is added
    after the reduction so it is not double-counted.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        input_is_parallel: bool = True,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.input_is_parallel = input_is_parallel
        self.tp_rank = rank()
        self.tp_world_size = world_size()
        self.local_in_features = in_features // self.tp_world_size
        if in_features % self.tp_world_size != 0:
            raise ValueError("in_features must divide evenly across tensor-parallel ranks.")

        self.weight = nn.Parameter(torch.empty(out_features, self.local_in_features))
        self.bias = nn.Parameter(torch.empty(out_features)) if bias else None
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in = self.in_features
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply this rank's input-feature shard and sum partial outputs.

        When `input_is_parallel=True`, `x` is already the local hidden shard from
        a previous column-parallel layer. When it is false, this layer first
        scatters a replicated input. Each rank computes a partial `[B, T,
        out_features]` result, then an all-reduce sums those partials into the
        full dense-layer output on every rank.
        """
        if self.input_is_parallel:
            local_x = x
        else:
            local_x = scatter_to_tensor_parallel_region(x)
        y = F.linear(local_x, self.weight, None)
        y = reduce_from_tensor_parallel_region(y)
        if self.bias is not None:
            y = y + self.bias
        return y


@torch.no_grad()
def copy_linear_to_column_parallel(src: nn.Linear, dst: ColumnParallelLinear) -> None:
    """Copy the output-feature slice of a dense `nn.Linear` into this rank."""
    start, end = _slice_bounds(src.out_features, dst.tp_rank, dst.tp_world_size)
    dst.weight.copy_(src.weight[start:end, :])
    if src.bias is not None and dst.bias is not None:
        dst.bias.copy_(src.bias[start:end])


@torch.no_grad()
def copy_linear_to_row_parallel(src: nn.Linear, dst: RowParallelLinear) -> None:
    """Copy the input-feature slice of a dense `nn.Linear` into this rank."""
    start, end = _slice_bounds(src.in_features, dst.tp_rank, dst.tp_world_size)
    dst.weight.copy_(src.weight[:, start:end])
    if src.bias is not None and dst.bias is not None:
        dst.bias.copy_(src.bias)
