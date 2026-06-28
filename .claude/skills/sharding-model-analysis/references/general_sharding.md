# General sharding strategies

Consult these before designing a strategy:
- Megatron-style tensor parallelism - https://arxiv.org/abs/1909.08053
- ZeRO sharding - https://arxiv.org/abs/1910.02054

Short summary so you know which to reach for; read the papers for the full picture.

## Megatron-style tensor parallelism (the default here)

Split the *weights* of a matmul pair across devices so the activation between them is
never gathered - **one all-reduce per pair**.

- **Column-parallel** (first matmul): shard the weight's *output* dim. Each device
  computes a slice of the output. No collective.
- **Row-parallel** (second matmul): shard the weight's *input* dim. Each device
  consumes its matching activation slice and produces a partial sum → **one all-reduce**
  to combine.

Example - an MLP `up` then `down` (PyTorch `nn.Linear` weight is `[out, in]`):
```
up.weight    ("model", None)   # column-parallel: shard output dim, no CCL
down.weight  (None, "model")   # row-parallel:    shard input dim → all-reduce after
```
Attention is the same template: `to_q/k/v` column-parallel, `to_out` row-parallel.
See [shardy_sharding.md](shardy_sharding.md) for the exact annotation syntax and
[ccl_cheatsheet.md](ccl_cheatsheet.md) for the collective each side emits.

## ZeRO sharding

A *data-parallel* memory optimization (primarily training-time): replicate compute
across ranks but shard what each rank *stores* - optimizer states (stage 1),
+ gradients (stage 2), + parameters (stage 3). It cuts per-device memory without
changing the math, paying an all-gather to reassemble parameters when needed.

Reach for it when the bottleneck is **memory under data parallelism**, not when you
need to split a single large layer's compute - that's Megatron's job. The two compose
(Megatron TP within a node, ZeRO/DP across nodes).
