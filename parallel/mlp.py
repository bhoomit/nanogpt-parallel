import torch.nn as nn

from model import MLP
from parallel.linear import (
    ColumnParallelLinear,
    RowParallelLinear,
    copy_linear_to_column_parallel,
    copy_linear_to_row_parallel,
)


class TensorParallelMLP(nn.Module):
    """nanoGPT MLP implemented with the canonical CPL -> activation -> RPL pair.

    The previous minimal commit wrote tensor parallelism directly in this MLP.
    This robust version names the two reusable pieces:

    - `ColumnParallelLinear`: split the expanded hidden output of `c_fc`.
    - `RowParallelLinear`: consume the hidden shard and all-reduce partial output.
    """

    def __init__(self, config):
        super().__init__()
        self.c_fc = ColumnParallelLinear(
            config.n_embd,
            4 * config.n_embd,
            bias=config.bias,
            gather_output=False,
        )
        self.gelu = nn.GELU()
        self.c_proj = RowParallelLinear(
            4 * config.n_embd,
            config.n_embd,
            bias=config.bias,
            input_is_parallel=True,
        )
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        """Run the tensor-parallel MLP while keeping the hidden activation sharded."""
        # CPL produces only this rank's slice of the expanded hidden dimension.
        x = self.c_fc(x)
        # GELU remains local because it does not mix features across shards.
        x = self.gelu(x)
        # RPL consumes the shard and all-reduces the partial residual output.
        x = self.c_proj(x)
        return self.dropout(x)


def copy_mlp_to_tensor_parallel(src: MLP, dst: TensorParallelMLP) -> None:
    """Copy dense nanoGPT MLP weights into their rank-local TP shards."""
    copy_linear_to_column_parallel(src.c_fc, dst.c_fc)
    copy_linear_to_row_parallel(src.c_proj, dst.c_proj)
