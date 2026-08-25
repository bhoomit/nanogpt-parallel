import sys
from pathlib import Path

import torch
import torch.distributed as dist

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from model import GPTConfig, MLP
from parallel.mlp import TensorParallelMLP, copy_mlp_to_tensor_parallel


def tiny_config() -> GPTConfig:
    return GPTConfig(
        block_size=8,
        vocab_size=32,
        n_layer=1,
        n_head=2,
        n_embd=4,
        dropout=0.0,
        bias=True,
    )


def main() -> None:
    dist.init_process_group(backend="gloo")
    try:
        if dist.get_world_size() != 2:
            raise RuntimeError(f"Expected 2 ranks, got {dist.get_world_size()}.")

        torch.manual_seed(0)
        config = tiny_config()
        reference = MLP(config)
        tp_mlp = TensorParallelMLP(config)

        # This is intentionally visible in the minimal commit: the TP module
        # starts empty, then copies this rank's dense nanoGPT weight shards.
        copy_mlp_to_tensor_parallel(reference, tp_mlp)

        x = torch.tensor([[[-1.0, 0.0, 1.0, 2.0], [2.0, 1.0, 0.0, -1.0]]])

        with torch.no_grad():
            reference_y = reference(x)
            tp_y = tp_mlp(x)

        torch.testing.assert_close(tp_y, reference_y, rtol=1e-6, atol=1e-6)
        if dist.get_rank() == 0:
            print("minimal TP MLP check passed")
    finally:
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
