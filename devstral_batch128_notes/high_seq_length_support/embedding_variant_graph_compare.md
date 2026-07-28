# Embedding sharding variant — graph efficiency comparison

Compared two decoder-region slices from TTNN IR dumps on a 4×8 mesh (DP=4 on cluster_axis 0, TP=8 on cluster_axis 1):

- **Slice A (OG):** `devstral_dptp_test_trace_off_OG.log` L7229–7589. Embedding weight `(None,"model")` → hidden sharded 8-way on TP (`131072x1536`); with the forward hook.
- **Slice B (no-hook):** `devstral_dptp_test_trace_off_no_hook_forward_shard.log` L7247–7622. Embedding weight `(None,"batch")` → hidden sharded 4-way on DP (`131072x3072`); forward hook removed.

**Both slices cover two full decoder layers.** The ONLY structural difference is the embedding/first-all-gather region. Everything downstream (matmuls, all_reduces, KV-cache partitions, attention, MLP) is byte-for-byte identical.

---

## 1. CCL ops

### Embedding region (THE difference)

| | Slice A (OG, TP-sharded weight) | Slice B (no-hook, DP-sharded weight) |
|---|---|---|
| Index gather | — | `all_gather` **DP** (axis0), dim1, ui32 `1x4096 → 1x16384` (~64 KB, negligible) |
| Embedding output | `4096x1536` bf16 (32 tok local, 1536 hid shard) | `16384x3072` bf16 (all 128 tok local, 3072 hid shard) |
| Token gather | `all_gather` **DP** (axis0), dim0, **bf16 `32x128x1536 → 128x128x1536`** (~25M elem ≈ **50 MB**) | — |
| Hidden gather | `all_gather` **TP** (axis1), dim2, bf16 `128x128x1536 → 128x128x12288` (201M elem out, **8-way**) | `all_gather` **DP** (axis0), dim2, bf16 `128x128x3072 → 128x128x12288` (201M elem out, **4-way**) |
| Re-scatter | `mesh_partition` DP axis0 dim0 `128x128x12288 → 32x128x12288` | `mesh_partition` DP axis0 dim0 `128x128x12288 → 32x128x12288` (identical) |

### Rest of the graph (IDENTICAL in both)

- **4× `all_reduce`** on **TP** (cluster_axis 1), `sum`, `1x1x4096x12288` bf16 — attn-out + MLP-out reductions (2 per layer).
- **4× `mesh_partition`** on **TP** (cluster_axis 1), dim1, `78336x8x32x128 → 78336x1x32x128` — KV-cache partitions.
- **0** `point_to_point`, **0** `reduce_scatter`, **0** `collective_permute` in either slice.

Totals: A = 2 all_gather / 5 mesh_partition / 4 all_reduce. B = 2 all_gather / 5 mesh_partition / 4 all_reduce (same op *counts*, but A's second all_gather is a 50 MB bf16 gather whereas B's is a 64 KB index gather).

## 2. Matmul ops — IDENTICAL

Every matmul feeds the **same** `4096x12288` per-device activation into the **same** weight shards, in both slices:

| Role | M×K · K×N (per device) | A / B |
|---|---|---|
| QKV fused proj | `4096x12288` · `12288x1792` → `4096x1792` | same |
| o_proj | `4096x1536` · `12288x1536`ᵀ → `4096x12288` | same |
| gate (silu) / up | `4096x12288` · `3584x12288`ᵀ → `4096x3584` | same |
| down | `4096x3584` · `12288x3584`ᵀ → `4096x12288` | same |

**The sharding change does not alter local matmul sizes.** Both variants converge to the identical `4096x12288` activation layout before the first matmul, so all 10 matmuls per slice are dimensionally identical.

## 3. Extra ops tied to the sharding difference

- **B has +2 `to_layout` and +1 `reshape`** — layout conversions on the tiny ui32 index tensor (`1x4096` / `16384`) around its pre-embedding index all_gather. Trivial (tiny integer tensors).
- **A has the extra 50 MB bf16 DP token all_gather** (`32x128x1536 → 128x128x1536`) that B does not.
- Everything else (slices, permutes, typecasts, deallocates, paged_fill_cache, RoPE) matches 1:1. Deallocate count differs by +3 in B purely from its 3 extra index-path ops.

---

## Verdict

**B (`(None,"batch")` + no hook) is the cheaper variant on communication; A is cheaper only on embedding-table memory.**

**B is cheaper because:**
1. It **eliminates A's ~50 MB bf16 DP token all_gather** entirely, replacing it with a ~64 KB ui32 *index* all_gather done *before* the embedding lookup. This is the decisive, unambiguous win — B gathers integer indices instead of wide bf16 hidden states.
2. It **trades the TP hidden all_gather for a DP hidden all_gather** — `cluster_axis 1 (8-way)` → `cluster_axis 0 (4-way)`. Same 201M-element output, but fewer hops per device (~151M vs ~176M received/device, ≈1.15×) *and* it moves the big gather **off the TP axis** (which is already carrying all 4 `all_reduce`s) onto the lightly-used DP axis — better axis balance.

Net: B receives ~20–25% less data per device in the embedding path and de-congests the TP axis.

**A is cheaper because (the tradeoff to flag):**
- A's embedding weight shard is **half the width** (`131072x1536` vs B's `131072x3072`) → **~2× less embedding-table memory per device**, and a smaller local lookup (`4096x1536` vs `16384x3072`). The lookup is a cheap gather, so the compute cost is negligible; the memory footprint is the real cost B pays.

**Bottom line:** does B trade the TP all_gather for a DP all_gather and drop overhead? **Yes** — direct TP→DP swap of the hidden gather, plus removal of the 50 MB bf16 token gather, at the price of a 2× wider embedding shard and 3 trivial extra layout ops. No point_to_point is introduced. Matmuls are identical. **B is the more efficient graph for this region.**
