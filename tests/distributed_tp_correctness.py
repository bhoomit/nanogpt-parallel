import sys
from pathlib import Path

import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from model import GPTConfig, MLP
from parallel.distributed import destroy_distributed, init_distributed, rank0_print, rank, world_size
from parallel.linear import (
    ColumnParallelLinear,
    RowParallelLinear,
    copy_linear_to_column_parallel,
    copy_linear_to_row_parallel,
)
from parallel.mappings import scatter_to_tensor_parallel_region
from parallel.mlp import TensorParallelMLP, copy_mlp_to_tensor_parallel


def _slice_bounds(size: int) -> tuple[int, int]:
    shard = size // world_size()
    start = rank() * shard
    return start, start + shard


def _config() -> GPTConfig:
    return GPTConfig(
        block_size=8,
        vocab_size=32,
        n_layer=1,
        n_head=2,
        n_embd=4,
        dropout=0.0,
        bias=True,
    )


def test_column_parallel_linear_forward_backward() -> None:
    torch.manual_seed(11)
    ref = nn.Linear(4, 8, bias=True)
    tp = ColumnParallelLinear(4, 8, bias=True, gather_output=True)
    copy_linear_to_column_parallel(ref, tp)

    x_ref = torch.randn(2, 3, 4, requires_grad=True)
    x_tp = x_ref.detach().clone().requires_grad_(True)
    grad = torch.randn(2, 3, 8)

    y_ref = ref(x_ref)
    y_tp = tp(x_tp)
    torch.testing.assert_close(y_tp, y_ref, rtol=1e-6, atol=1e-6)

    (y_ref * grad).sum().backward()
    (y_tp * grad).sum().backward()

    start, end = _slice_bounds(ref.out_features)
    torch.testing.assert_close(x_tp.grad, x_ref.grad, rtol=1e-6, atol=1e-6)
    torch.testing.assert_close(tp.weight.grad, ref.weight.grad[start:end], rtol=1e-6, atol=1e-6)
    torch.testing.assert_close(tp.bias.grad, ref.bias.grad[start:end], rtol=1e-6, atol=1e-6)


def test_row_parallel_linear_forward_backward() -> None:
    torch.manual_seed(22)
    ref = nn.Linear(8, 4, bias=True)
    tp = RowParallelLinear(8, 4, bias=True, input_is_parallel=False)
    copy_linear_to_row_parallel(ref, tp)

    x_ref = torch.randn(2, 3, 8, requires_grad=True)
    x_tp = x_ref.detach().clone().requires_grad_(True)
    grad = torch.randn(2, 3, 4)

    y_ref = ref(x_ref)
    y_tp = tp(x_tp)
    torch.testing.assert_close(y_tp, y_ref, rtol=1e-6, atol=1e-6)

    (y_ref * grad).sum().backward()
    (y_tp * grad).sum().backward()

    start, end = _slice_bounds(ref.in_features)
    torch.testing.assert_close(x_tp.grad, x_ref.grad, rtol=1e-6, atol=1e-6)
    torch.testing.assert_close(tp.weight.grad, ref.weight.grad[:, start:end], rtol=1e-6, atol=1e-6)
    torch.testing.assert_close(tp.bias.grad, ref.bias.grad, rtol=1e-6, atol=1e-6)


def test_tp_mlp_forward_backward_matches_nanogpt_mlp() -> None:
    torch.manual_seed(33)
    config = _config()
    ref = MLP(config)
    tp = TensorParallelMLP(config)
    copy_mlp_to_tensor_parallel(ref, tp)

    x_ref = torch.randn(2, 3, config.n_embd, requires_grad=True)
    x_tp = x_ref.detach().clone().requires_grad_(True)
    grad = torch.randn(2, 3, config.n_embd)

    y_ref = ref(x_ref)
    y_tp = tp(x_tp)
    torch.testing.assert_close(y_tp, y_ref, rtol=1e-6, atol=1e-6)

    (y_ref * grad).sum().backward()
    (y_tp * grad).sum().backward()

    fc_start, fc_end = _slice_bounds(ref.c_fc.out_features)
    proj_start, proj_end = _slice_bounds(ref.c_proj.in_features)
    torch.testing.assert_close(x_tp.grad, x_ref.grad, rtol=1e-6, atol=1e-6)
    torch.testing.assert_close(tp.c_fc.weight.grad, ref.c_fc.weight.grad[fc_start:fc_end], rtol=1e-6, atol=1e-6)
    torch.testing.assert_close(tp.c_fc.bias.grad, ref.c_fc.bias.grad[fc_start:fc_end], rtol=1e-6, atol=1e-6)
    torch.testing.assert_close(tp.c_proj.weight.grad, ref.c_proj.weight.grad[:, proj_start:proj_end], rtol=1e-6, atol=1e-6)
    torch.testing.assert_close(tp.c_proj.bias.grad, ref.c_proj.bias.grad, rtol=1e-6, atol=1e-6)


def test_scatter_requires_even_last_dimension() -> None:
    try:
        scatter_to_tensor_parallel_region(torch.randn(2, 3))
    except ValueError as exc:
        assert "Cannot split last dimension" in str(exc)
    else:
        raise AssertionError("scatter should reject non-divisible last dimensions")


def main() -> None:
    init_distributed(expected_world_size=2)
    try:
        test_column_parallel_linear_forward_backward()
        test_row_parallel_linear_forward_backward()
        test_tp_mlp_forward_backward_matches_nanogpt_mlp()
        test_scatter_requires_even_last_dimension()
        rank0_print("distributed TP correctness tests passed")
    finally:
        destroy_distributed()


if __name__ == "__main__":
    main()
