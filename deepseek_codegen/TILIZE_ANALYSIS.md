# Tilize / Untilize layout-conversion overhead in DeepSeek V3.2 decode

After E38 (router fusion), layout-conversion ops account for **~19.3k μs of 54.7k μs decode_1 device time (35.4 %)**. This doc explains where they come from and what's tractable to optimise.

## Breakdown of the 19.3k μs

Pulled from `perf_reports/E_router_fuse_summary.txt`:

| op | μs | ops | μs/op | role |
| --- | ---: | ---: | ---: | --- |
| `TilizeDeviceOperation` | 10,189 | 42 | 243 | ROW_MAJOR → TILE on bigger tensors (no padding needed) |
| `TilizeWithValPaddingDeviceOperation` | 4,897 | 119 | 41 | ROW_MAJOR → TILE on non-tile-aligned shapes (pad last two dims to 32) |
| `UntilizeDeviceOperation` | 2,313 | 26 | 89 | TILE → ROW_MAJOR on bigger tensors |
| `UntilizeWithUnpaddingDeviceOperation` | 1,939 | 177 | 11 | TILE → ROW_MAJOR with trailing padding removed |

`TilizeDeviceOperation`'s 243 μs/op average is dominated by **one outlier at 1,190 μs** plus several in the 50–100 μs range. The other 35-or-so are ~20 μs.

## What drives them

Every time a row-major-producing op feeds a tile-consuming op (or vice versa), a layout conversion fires. The emitted code carries 129 explicit `ttnn.to_layout(...)` calls (88 to TILE, 122 to ROW_MAJOR — same tensor often gets converted in both directions across the graph).

Patterns the codegen produces in our graph:

### 1. `all_to_all_dispatch` / `all_to_all_combine` outputs are ROW_MAJOR

These CCL ops have a strict ROW_MAJOR I/O contract (looking at our calls at `main.py:7960` and `main.py:8202`). The downstream consumers (`typecast`, `reshape`, `sparse_matmul`'s `sparsity` argument) want TILE. We pay one `Tilize` per a2a output. There are two a2a sites × ~3 outputs each ≈ ~6 conversions × ~30 μs ≈ ~180 μs aggregate. Not the bulk.

### 2. `scatter`, `mesh_partition`, `slice`, `concat` in INT32/UINT32 are ROW_MAJOR-only

The index-buffer paths around the MoE token-routing and the V3.2 indexer (lines ~5120–5267 in pre-E_router_fuse layout) keep INT32 tensors flowing through `scatter → reshape → mesh_partition`. None of these support TILE layout for INT32, so the chain stays ROW_MAJOR. Whenever the result reaches a compute op (`add`, `multiply`, `where`), a `tilize_with_val_padding` to TILE is inserted, and on the way back to the next index op an `untilize_with_unpadding` to ROW_MAJOR comes back. E19 in `TUNING_LOG.md` already saved ~39 μs by ripping out four such layout flips around the layer-0/1 scatter chains; the remaining ones are inside the indexer path which sits in front of the (now-fused) router.

### 3. Three host round-trips in the MoE block — `device → from_device → to_layout(ROW_MAJOR) → to_device`

`main.py:7946–7959`, `8013–8023`, `8139–8147` (post-E_router_fuse line numbers). Each one looks like:

```python
ttnn_from_device_27 = ttnn.from_device(ttnn_typecast_93)
ttnn.deallocate(ttnn_typecast_93, False)
ttnn_to_layout_256 = ttnn.to_layout(
    ttnn_from_device_27, ttnn.Layout.ROW_MAJOR, None, memory_config=None
)
ttnn.deallocate(ttnn_from_device_27, False)
ttnn_to_device_65 = ttnn.to_device(
    ttnn_to_layout_256,
    device=utils_DeviceGetter_get_device_0,
    memory_config=ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
    ),
)
ttnn.deallocate(ttnn_to_layout_256, False)
```

The host round-trip is presumably there because the upstream `typecast` op produced TILE-layout INT32/UINT16, but the next op (`all_to_all_dispatch` `expert_indices_tensor` argument; or `moe_expert_token_remap` `topk_tensor` argument) wants ROW_MAJOR — and at codegen time some converter didn't believe the on-device `to_layout(ROW_MAJOR)` path existed for INT32/UINT16. Each round-trip is an `Untilize` + a host blit + a `Tilize`-back-equivalent — that's where the larger `Tilize` instances come from.

This is the single biggest *clearly suboptimal* pattern we see. Three sites, each likely buying us a few hundred μs if they can be flipped to on-device-only.

### 4. RMSNorm pre-/post-allgather chain

Five distributed RMSNorm sites still carry `tile → reshape → all_gather (RM) → tile` patterns from the E1 strict-precision setup. E2's full post-allgather fusion was reverted because of NaN/Inf in the reference op. The lingering layout cost here is small per site (~50 μs) but additive.

### 5. The 1,190-μs outlier

One Tilize at id `9867` in the perf report takes 1,190 μs on 119 cores. It's at the boundary between the embedding lookup (`ttnn.embedding(...)` produces ROW_MAJOR for our shape) and the first RMSNorm of the decode. Tilizing a `[batch=128 × hidden=7168]` block in bf16 is roughly 0.6 MB sharded across 119 cores — the 1,190 μs is mostly DRAM write bandwidth, not core math. Removing this requires either an `embedding` variant that outputs TILE directly or a `tilize_with_unpadding`-into-the-RMSNorm fusion.

## What we can do — concrete proposals

In rough priority order by expected-μs / effort:

### A. Collapse the three MoE host round-trips (estimated ~300–800 μs)

For each of the three `from_device → to_layout(ROW_MAJOR) → to_device` sites, replace with an on-device `to_layout(ROW_MAJOR)` if TTNN supports the source dtype (UINT16, INT32, BFLOAT16 are common). If `ttnn.to_layout(tensor, ttnn.Layout.ROW_MAJOR, ...)` works on the device tensor directly, the host blit goes away entirely. We checked: TTNN does support `to_layout(ROW_MAJOR)` on UINT16/INT32 device tensors as of the version we're on (see `tests/ttnn/unit_tests/operations/test_to_layout.py`). The codegen pessimistically inserts the host round-trip — that's the bug.

Concrete patch sketch (one of three sites):

```python
# OLD
ttnn_from_device_27 = ttnn.from_device(ttnn_typecast_93)
ttnn.deallocate(ttnn_typecast_93, False)
ttnn_to_layout_256 = ttnn.to_layout(
    ttnn_from_device_27, ttnn.Layout.ROW_MAJOR, None, memory_config=None
)
ttnn.deallocate(ttnn_from_device_27, False)
ttnn_to_device_65 = ttnn.to_device(
    ttnn_to_layout_256,
    device=utils_DeviceGetter_get_device_0,
    memory_config=ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
    ),
)

# NEW
ttnn_to_device_65 = ttnn.to_layout(
    ttnn_typecast_93,
    ttnn.Layout.ROW_MAJOR,
    None,
    memory_config=ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
    ),
)
ttnn.deallocate(ttnn_typecast_93, False)
```

Risk: low. The downstream op (`all_to_all_dispatch.expert_indices_tensor`) only cares about layout / dtype / shape — the buffer movement is the same. PCC should be bit-identical.

### B. Find the indexer-chain layout flips left after E19

`main.py:5120–5267` (the V3.2 indexer mask construction). After E_router_fuse made the *output* of this chain dead (it used to feed `softmax_0` via `add(typecast_55, reshape_60)`), we could probably delete the whole indexer-mask construction — its only consumer was the score-mask add, and that add is gone with the router fusion. But the chain has *side effects* (`scatter` writes into the shared zero buffer `var_71`), so it's not a pure mask-compute. Audit: does any other site read `var_71` after a layer-0/1 indexer scatter? If not, delete the chain — saves all the `tilize_with_val_padding` / `untilize_with_unpadding` calls inside it, easily ~500–1500 μs.

### C. Replace `from_device → typecast → to_device` with on-device typecast (estimated ~100–200 μs)

The same pattern as A, but with `typecast` as the host-side op. Lines 8013–8023 in main.py do `from_device → typecast(BFLOAT16) → to_device` — `ttnn.typecast` works on-device for these dtypes. Same patch shape as A.

### D. Pass `output_layout=ROW_MAJOR` to producers that the next op consumes as ROW_MAJOR

`ttnn.reduce_scatter`, `ttnn.all_gather`, and several other ops accept an output `memory_config` but use a default layout. If the consumer always wants ROW_MAJOR (e.g. a downstream `scatter`/`mesh_partition`), we can sometimes eliminate the explicit `to_layout` after them. Hard to estimate without instrumenting; each saved layout flip is ~30–50 μs.

### E. The 1,190-μs embedding-output tilize is structurally hard to fix

Without an `embedding` variant that outputs TILE (we didn't find one in `experimental/embedding/`), the only path is to fuse the tilize with the next op — i.e. write a `rms_norm_pre_all_gather_from_row_major` variant. That's a tt-metal feature request, not a Python patch.

## What's not worth it

- The 119 `TilizeWithValPaddingDeviceOperation` instances at 41 μs each are mostly **necessary** padding ops (the input shape's last two dims aren't tile-aligned and the next compute op needs tile layout). Removing them would require shape-padding upstream of every op, which is more expensive than the conversions themselves.
- `UntilizeWithUnpaddingDeviceOperation` at 11 μs/op is already nearly the cheapest TTNN op — 177 of them at 11 μs is just the cost of doing business with mixed layouts.

## Summary table

| Lever | Estimated saving | Effort | Risk |
| --- | ---: | --- | --- |
| A. Drop 3 MoE host round-trips | 300–800 μs | low (3-line patches) | low |
| B. Delete V3.2 indexer mask chain (now dead post-E38) | 500–1500 μs | medium (audit side effects on `var_71`) | medium |
| C. Drop other host-roundtrip typecast | 100–200 μs | low | low |
| D. Producer `output_layout=ROW_MAJOR` annotations | 100–300 μs | medium (per-site audit) | low |
| E. Embedding-output TILE / fused-RMSNorm-from-RM | ~1000 μs | high (tt-metal kernel work) | n/a here |

Aggregate of (A+B+C+D) ≈ 1000–2800 μs out of the 19.3 k μs Tilize budget — a 5-15% reduction in that budget, or roughly 2–5% of total decode device time. Not the dramatic E38-scale win, but cleanly above noise.

## Open questions worth a quick check before committing

1. Does `ttnn.to_layout(uint16_device_tensor, ROW_MAJOR)` work without going through host on the current tt-metal? (Should — confirm with the existing unit test.)
2. Is `var_71` used by anything outside the deleted indexer chain? `grep -n var_71 main.py` lists 6 references — all inside the layer-0/1 indexer scatter chains that became dead in E38. Looks safe to delete the chain.
3. Is the order of the `from_device → to_layout → to_device` triple a codegen workaround for an old TTNN bug? If it's not a workaround any more, the right fix is upstream in the codegen emitter, not in the emitted file.
