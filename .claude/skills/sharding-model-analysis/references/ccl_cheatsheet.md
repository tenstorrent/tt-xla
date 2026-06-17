# CCL cheat-sheet - which collective each sharding pattern emits

Use this for the CCL accounting in your analysis. It maps a sharding pattern to the
collective the Shardy partitioner inserts, at the *logical* level (all-reduce /
all-gather / reduce-scatter).

> **This can change.** Partitioner behavior and tt-mlir lowering evolve. Treat this as
> a starting point and confirm against the exported IR (phase 7) and current tt-mlir -
> not as a fixed contract.

## Pattern → collective

| Sharding pattern | Collective inserted | Source |
|---|---|---|
| Column-parallel matmul (shard weight *output* dim) | none - output stays sharded | Megatron §3 (arxiv 1909.08053); [video_dit.md](video_dit.md), [video_vae.md](video_vae.md) |
| Row-parallel matmul (shard weight *input*/contraction dim) | **1 all-reduce** (local output is a partial sum) | Megatron §3; [video_vae.md](video_vae.md), [video_dit.md](video_dit.md) |
| Reduction (norm / softmax stats) over a sharded axis | **1 all-reduce** of the reduced stats (small - the axis is collapsed before the exchange) | [video_vae.md](video_vae.md) ("norm2"), [video_dit.md](video_dit.md) ("norm_q/k stats AR") |
| Attention over a sequence-sharded activation needing full K/V | **all-gather** K and V along the sequence axis | [video_dit.md](video_dit.md) ("AG of K, V along SP") |
| Resharding sharded → replicated | **all-gather** | [video_dit.md](video_dit.md) (exit at `proj_out`) |

## One decomposition identity

`all_reduce ≡ reduce_scatter + all_gather` - 2 collectives. This is the standard
all-reduce decomposition and is stated in [video_vae.md](video_vae.md)
("all_reduce, which decomposes as reduce_scatter + all_gather - 2 ccls"). Handy when
weighing a row-parallel all-reduce against a column-parallel result that only needs a
single all-gather downstream.

## Verifying op names in the IR

The only collective op name confirmed in this repo's IR tests is `ttir.all_gather`
(`tests/filecheck/sharding_constraints.ttir.mlir`). Do **not** assume other TTIR/TTNN
op spellings - read the exported TTIR/TTNN for your model to see exactly what was
emitted.
