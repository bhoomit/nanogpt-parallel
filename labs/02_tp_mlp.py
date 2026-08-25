import argparse
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from model import GPTConfig, MLP
from parallel.distributed import destroy_distributed, init_distributed, ordered_print, rank0_print
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
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", default=None, help="Default: gloo for CPU-friendly tiny-tensor labs.")
    parser.add_argument("--trace", action="store_true", help="Print tensors before and after TP communication.")
    args = parser.parse_args()

    init_distributed(backend=args.backend, expected_world_size=2)
    torch.manual_seed(0)

    config = tiny_config()
    reference = MLP(config)
    tp_mlp = TensorParallelMLP(config)
    copy_mlp_to_tensor_parallel(reference, tp_mlp)

    x = torch.tensor(
        [
            [[-1.0, 0.0, 1.0, 2.0], [2.0, 1.0, 0.0, -1.0]],
        ]
    )
    with torch.no_grad():
        reference_y = reference(x)
        local_hidden = tp_mlp.gelu(tp_mlp.c_fc(x))

    if args.trace:
        ordered_print(
            "before row-parallel projection",
            c_fc_weight_shard=tp_mlp.c_fc.weight,
            c_fc_bias_shard=tp_mlp.c_fc.bias,
            local_hidden=local_hidden,
            c_proj_weight_shard=tp_mlp.c_proj.weight,
            c_proj_bias=tp_mlp.c_proj.bias,
        )

    with torch.no_grad():
        tp_y = tp_mlp(x)

    if args.trace:
        ordered_print(
            "after row-parallel all_reduce",
            tp_output=tp_y,
            reference_single_rank_output=reference_y,
            max_abs_diff=(tp_y - reference_y).abs().max(),
        )

    torch.testing.assert_close(tp_y, reference_y, rtol=1e-6, atol=1e-6)
    rank0_print(
        "\nTP-MLP equivalence passed. The nanoGPT MLP c_fc is column-parallel, "
        "c_proj is row-parallel, and the row-parallel all_reduce reconstructs "
        "the same output as the original single-process MLP."
    )
    destroy_distributed()


if __name__ == "__main__":
    main()
