# nanoGPT With Parallelism, From First Principles

[![tests](https://github.com/OWNER/REPO/actions/workflows/tests.yml/badge.svg)](https://github.com/OWNER/REPO/actions/workflows/tests.yml)

One small mechanism at a time.

This fork keeps [nanoGPT](https://github.com/karpathy/nanoGPT) as the readable
transformer skeleton and adds distributed parallelism beside it, one concept at
a time.

The goal is not to build a production trainer. The goal is to make the mechanics
of transformer parallelism as legible as nanoGPT made a GPT training loop.

By the end of the series, the goal is to converge to a small but runnable
parallel version of nanoGPT: tensor parallel MLPs and attention, pipeline stages,
sequence/context parallel attention, expert-parallel MLPs, and sharded
checkpointing. Each piece should be understandable from the tensor math and
checked against the original nanoGPT component wherever possible.

Terminology note: NVIDIA commonly uses "5D parallelism" for TP, PP, DP, CP, and
EP. In this series, DP is treated as the familiar outer replication axis, while
the hands-on focus is TP, PP, SP, CP, EP, and sharded checkpointing inside a
nanoGPT-shaped model.

First-principles here means:

```text
start from the original nanoGPT equation
use tiny tensors
slice weights by hand
compute each rank's local result
show exactly where communication is required
then package the idea into reusable modules
```

This is an educational fork built on Karpathy's open-source nanoGPT, not an
official nanoGPT project.

## Related Reading

- [NVIDIA Megatron Core Parallelism Strategies Guide](https://docs.nvidia.com/megatron-core/developer-guide/latest/user-guide/parallelism-guide.html): production terminology and configuration guidance for TP, PP, CP, and EP.
- [NVIDIA NeMo AutoModel SFT/PEFT recipe](https://docs.nvidia.com/nemo/automodel/recipes-e2e-examples/sft-peft): a practical distributed-training config table covering DP, TP, PP, CP, and EP.
- [NVIDIA NeMo Context Parallelism](https://docs.nvidia.com/nemo-framework/user-guide/25.02/longcontext/contextparallel.html): a focused reference for long-context CP.

## Launch The Notebook

These links will work after this fork is pushed to a public GitHub repo. Replace
`OWNER/REPO` with the published repository path.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/OWNER/REPO/blob/main/notebooks/01_tp_mlp_from_first_principles.ipynb)
[![Open In Kaggle](https://kaggle.com/static/images/open-in-kaggle.svg)](https://www.kaggle.com/notebooks/welcome?src=https://github.com/OWNER/REPO/blob/main/notebooks/01_tp_mlp_from_first_principles.ipynb)
[![Launch Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/OWNER/REPO/main?labpath=notebooks/01_tp_mlp_from_first_principles.ipynb)

Free runtime notes:

```text
Colab: easiest path for most readers.
Kaggle: use the badge or File -> Load from URL with the GitHub notebook URL.
Binder: useful for public demos, but PyTorch startup can be slower.
```

## Quick Parallelism Cheat Sheet

| Transformer piece | Parallel type | Communication after op? |
| --- | --- | --- |
| MLP `c_fc` / up projection | Column parallel | No gather before GELU |
| MLP activation | Local elementwise | None |
| MLP `c_proj` / down projection | Row parallel | `all_reduce(SUM)` |
| Attention QKV projection | Column parallel over heads | Usually no gather before attention |
| Attention per-head compute | Local head parallel | None for TP |
| Attention output projection | Row parallel | `all_reduce(SUM)` |
| Plain LayerNorm | Replicated | None |
| Sequence-parallel boundary | Sequence parallel | `all_gather` or `reduce_scatter` |
| LoRA / small adapters | Framework-specific | Often replicated for SP; TP may mirror the base layer |

Rule of thumb: shard the big matrices and expanded activations when savings
dominate communication; replicate tiny trainable updates like many LoRA adapters
when communication would dominate the memory saved. A TP'd base MLP does not
force one universal LoRA strategy: frameworks may replicate adapter parameters
under SP, mirror the base layer's TP layout, or disable specific communication
paths when the adapter is too small to justify them.

Run:

```bash
torchrun --standalone --nproc-per-node=2 labs/02_tp_mlp.py --trace
```

Current scope: this section validates forward and backward equivalence for
tensor-parallel linear layers and the nanoGPT MLP. Optimizer-state ownership and
checkpoint mappings come later in the series.

Run the correctness tests:

```bash
python -m pytest tests -q
```

## Table Of Contents

These entries start as a roadmap and will become links as each section is
published.

| # | Section | First-principles implementation |
| ---: | --- | --- |
| 1 | Why nanoGPT is the right skeleton | Keep original `model.py` as reference |
| 2 | Tensor-parallel MLP | Shard `c_fc` and `c_proj`; match nanoGPT MLP |
| 3 | Column-parallel linear layers | Trace scatter/gather decisions with tiny tensors |
| 4 | Row-parallel linear layers | Trace partial outputs and all-reduce |
| 5 | Tensor-parallel attention heads | Shard QKV heads and output projection |
| 6 | Tensor-parallel transformer block | Compose TP attention and TP MLP |
| 7 | Pipeline parallelism | Split transformer blocks into stages |
| 8 | Pipeline microbatches | Show bubbles with a tiny schedule |
| 9 | Sequence parallelism | Shard token activations |
| 10 | Context parallel attention | Move long-context K/V blocks |
| 11 | Expert parallelism | Route tokens to expert MLPs |
| 12 | All-to-all for MoE | Trace dispatch and combine |
| 13 | Sharded checkpoints | Save and load rank-local shards |
| 14 | Putting the pieces together | Final map, limits, and tradeoffs |

## Original nanoGPT

The baseline remains Karpathy's nanoGPT. The original files are useful because
every parallel component can be checked against a compact single-process
reference. See [README_NANOGPT.md](README_NANOGPT.md) for the preserved original
project README.

## Note

I am building the series with Codex as a coding partner: using it to scaffold
experiments, run checks, and keep the repo, notebooks, and posts synchronized.
The aim of the series remains the distributed-systems ideas themselves; Codex is
part of the build process, not the subject.
