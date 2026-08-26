import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from parallel.distributed import destroy_distributed, init_distributed, ordered_print, rank, rank0_print
from parallel.mappings import (
    copy_to_tensor_parallel_region,
    gather_from_tensor_parallel_region,
    reduce_from_tensor_parallel_region,
    scatter_to_tensor_parallel_region,
)


def exercise_copy_to_region() -> None:
    """Show TP region entry: forward identity, backward all-reduce."""
    rank0_print("\nExercise 1: copy_to_tensor_parallel_region")
    x = torch.tensor([[1.0, 2.0]], requires_grad=True)

    # Forward is intentionally boring: every rank keeps the same replicated x.
    y = copy_to_tensor_parallel_region(x)

    # Backward is the point. Rank 0 contributes ones, rank 1 contributes twos.
    local_grad = torch.full_like(y, float(rank() + 1))
    (y * local_grad).sum().backward()

    ordered_print(
        "copy_to: forward identity, backward all_reduce",
        replicated_input=x.detach(),
        local_backward_contribution=local_grad,
        output=y.detach(),
        input_grad_after_backward=x.grad,
    )
    torch.testing.assert_close(x.grad, torch.full_like(x, 3.0))
    rank0_print("Explanation: copy_to does not move data in forward. It sums replicated input gradients in backward.")


def exercise_reduce_from_region() -> None:
    """Show TP region exit: forward all-reduce, backward identity."""
    rank0_print("\nExercise 2: reduce_from_tensor_parallel_region")
    partial = torch.tensor([[1.0, 2.0]], requires_grad=True) * float(rank() + 1)
    partial.retain_grad()

    # Forward sums rank-local partial outputs so every rank gets the same tensor.
    y = reduce_from_tensor_parallel_region(partial)
    y.sum().backward()

    ordered_print(
        "reduce_from: forward all_reduce, backward identity",
        local_partial_before_all_reduce=partial.detach(),
        output_after_all_reduce=y.detach(),
        partial_grad_after_backward=partial.grad,
    )
    torch.testing.assert_close(y, torch.tensor([[3.0, 6.0]]))
    torch.testing.assert_close(partial.grad, torch.ones_like(partial))
    rank0_print("Explanation: reduce_from is the forward all-reduce used after row-parallel partial outputs.")


def exercise_scatter_to_region() -> None:
    """Show activation sharding: forward split, backward gather."""
    rank0_print("\nExercise 3: scatter_to_tensor_parallel_region")
    x = torch.tensor([[0.0, 1.0, 2.0, 3.0]], requires_grad=True)

    # Forward splits the last dimension: rank 0 gets [0, 1], rank 1 gets [2, 3].
    y = scatter_to_tensor_parallel_region(x)

    # Backward gathers gradient shards back into the replicated input gradient.
    local_grad = torch.full_like(y, float(rank() + 1))
    (y * local_grad).sum().backward()

    ordered_print(
        "scatter_to: forward split, backward gather",
        replicated_input=x.detach(),
        local_shard_after_scatter=y.detach(),
        local_backward_contribution=local_grad,
        input_grad_after_backward=x.grad,
    )
    torch.testing.assert_close(x.grad, torch.tensor([[1.0, 1.0, 2.0, 2.0]]))
    rank0_print("Explanation: scatter_to really shards the activation in forward and rebuilds gradient shards in backward.")


def exercise_gather_from_region() -> None:
    """Show activation reconstruction: forward gather, backward split."""
    rank0_print("\nExercise 4: gather_from_tensor_parallel_region")
    local = torch.tensor([[0.0, 1.0]], requires_grad=True) + float(2 * rank())
    local.retain_grad()

    # Forward concatenates rank-local last-dimension shards into a replicated tensor.
    y = gather_from_tensor_parallel_region(local)

    # Backward splits the full output gradient back to the matching input shard.
    full_grad = torch.tensor([[10.0, 20.0, 30.0, 40.0]])
    (y * full_grad).sum().backward()

    ordered_print(
        "gather_from: forward gather, backward split",
        local_shard_before_gather=local.detach(),
        output_after_gather=y.detach(),
        full_backward_contribution=full_grad,
        local_grad_after_backward=local.grad,
    )
    expected_local_grad = torch.tensor([[10.0, 20.0]]) if rank() == 0 else torch.tensor([[30.0, 40.0]])
    torch.testing.assert_close(y, torch.tensor([[0.0, 1.0, 2.0, 3.0]]))
    torch.testing.assert_close(local.grad, expected_local_grad)
    rank0_print("Explanation: gather_from rebuilds the activation in forward and returns each rank's gradient slice in backward.")


def main() -> None:
    init_distributed(expected_world_size=2)
    try:
        rank0_print(
            "TP region functions are easiest to read as forward/backward pairs.\n"
            "Run with: torchrun --standalone --nproc-per-node=2 labs/03_tp_regions.py"
        )
        exercise_copy_to_region()
        exercise_reduce_from_region()
        exercise_scatter_to_region()
        exercise_gather_from_region()
        rank0_print("\nTP region lab passed.")
    finally:
        destroy_distributed()


if __name__ == "__main__":
    main()
