# TM-reduction plan after E41 — tracing every TilizeDeviceOperation to its producer

`TM` = tile-manipulation: `TilizeDeviceOperation`, `TilizeWithValPaddingDeviceOperation`, `UntilizeDeviceOperation`, `UntilizeWithUnpaddingDeviceOperation`.

After E41 (SDPA fusion), the per-iter decode_1 device-time breakdown for layout conversion is:

| Op | μs / iter | Op count |
| --- | ---: | ---: |
| `TilizeDeviceOperation` | 10,202 | 42 |
| `TilizeWithValPaddingDeviceOperation` | 4,879 | 119 |
| `UntilizeDeviceOperation` | 2,298 | 26 |
| `UntilizeWithUnpaddingDeviceOperation` | 1,955 | 179 |
| **TM total** | **19,334** | 366 |

The 366 TM ops per decode iteration are **38% of the 50,445 μs decode_1 budget**. Most are NOT visible as explicit `ttnn.to_layout(...)` calls in `main.py` (only 129 such calls exist, and many produce row-major output) — they're inserted by other ops when they receive an input in the wrong layout, or are auto-emitted by ops that always produce row-major output even though their consumer wants tile.

## Per-shape breakdown (sorted by per-iter device-time)

Pulled from `ops_perf_results_2026_05_21_18_15_42.csv`, grouped by `(op_code, input_shape, input_dtype, in_layout→out_layout)`, summed across calls inside `decode_1_start..decode_1_end`, then divided by 32 devices (because TM ops run on every chip in parallel, the *device* time is the max-across-devices, which is approximately the per-device time):

| Rank | Op | Shape (W×Z×Y×X) | dtype | dir | calls (32 dev × N/dev) | μs / iter | Producer guess |
| ---: | --- | --- | --- | --- | ---: | ---: | --- |
| 1 | Tilize | 1×8×7168×2048 | BF16 | →TILE | 128 | **4,902** | MoE expert weight broadcast / sparse_matmul input prep |
| 2 | Tilize | 1×1×129280×896 | BF16 | →TILE | 64 | **2,346** | LM-head weight tile |
| 3 | TilizeVal | 1×512×1×7168 | BF16 | →TILE | 32 | **1,642** | post-all_gather residual/hidden broadcast |
| 4 | Tilize | 1×8×2048×7168 | BF16 | →TILE | 32 | **1,557** | MoE expert weight (transpose pair of #1) |
| 5 | Untilize | 1×1×129280×896 | BF16 | →RM | 32 | **1,309** | LM-head output prep for argmax / from_device |
| 6 | TilizeVal | 1×1×1×256 | FP32 | →TILE | 2048 | **1,274** | routing-weight per-token broadcast scaffold (post-E38 leftover) |
| 7 | TilizeVal | 1×16384×32×2 | BF16 | →TILE | 32 | **1,167** | indexer / scatter mask broadcast |
| 8 | Tilize | 1×1×7168×2048 | BF16 | →TILE | 64 | **475** | MLA wkv_b1/wkv_b2 weight (small) |
| 9 | UntilizeVal | 1×1×32×16384 | FP32 | →RM | 2112 | **474** | indexer mask scatter prep |
| 10 | Untilize | 8×1×512×7168 | BF16 | →RM | 32 | **469** | sparse_matmul output → all_to_all_combine |
| 11 | UntilizeVal | 1×1×32×256 | FP32 | →RM | 2048 | **315** | routing-weight broadcast tail |
| 12 | Untilize | 8×1×128×7168 | BF16 | →RM | 32 | **306** | sparse_matmul output (split shape) |
| 13 | UntilizeVal | 32×16×32×32 | FP32 | →RM | 192 | **275** | MLA-attention small mask scatter |
| 14 | Tilize | 1×1×3072×1536 | BF16 | →TILE | 64 | **206** | MLP intermediate weight |
| 15 | UntilizeVal | 1×1×16384×32 | INT32 | →RM | 128 | **183** | indexer index buffer flatten |

**The top 5 entries alone account for 11,756 μs / 61 % of the entire TM budget.** They're all weight/activation tilizes around the big matmuls, the LM head, and the post-all_gather hidden broadcast.

## Classification by removability

### Class A — auto-inserted at matmul input boundaries (entries 1, 2, 4, 8, 14)

`TilizeDeviceOperation` shapes that match weight or large-activation matmul inputs: `1×8×7168×2048`, `1×1×129280×896`, `1×8×2048×7168`, `1×1×7168×2048`, `1×1×3072×1536`. **These are not visible as `ttnn.to_layout(...)` in main.py.** They appear because:

- `ttnn.matmul(activation, weight)` where `weight` is loaded from a const_eval that produced ROW_MAJOR → matmul prepends a Tilize on the weight input. This shouldn't happen because const_eval weights are already tilized at startup (see `main_const_eval_*`), but the matmul wrapper still emits a "make sure it's tile" call that the runtime fast-paths to a no-op when already tile. **Verify** with a trace: are these Tilize calls actually mapping to ROW_MAJOR→TILE conversions, or are they fast-path no-ops counted oddly?
- `ttnn.sparse_matmul`'s `sparsity` argument: the sparsity tensor flows through `matmul_29 → reshape → concat → all_gather → reshape → typecast → to_layout(ROW_MAJOR)` and arrives at `sparse_matmul` as ROW_MAJOR. `sparse_matmul` then tilizes it internally.

**Action:** for the top weight-tilize entries, add a one-shot `print(weight.layout())` inside main.py at the call site to confirm whether the weight is actually ROW_MAJOR. If it is, fix the const_eval to emit TILE. If it's already TILE, the Tilize call is misclassified perf-data and the actual cost is somewhere else.

### Class B — explicit `to_layout(ROW_MAJOR)` that feeds a tile-wanting consumer (entries 3, 5)

`1×512×1×7168` TilizeVal and `1×1×129280×896` Untilize are produced by ops with explicit ROW_MAJOR output: `all_gather`, `embedding`, `lm_head matmul output → from_device → to_device` triple. The consumer always wants TILE.

**Action — entry 3 (post-AG broadcast tilize, 1642 μs):** find the `all_gather` that produces `1×512×7168` shape (probably the hidden-state broadcast feeding sparse_matmul or matmul_29). If we can issue the all_gather with `memory_config` already in a tile-friendly layout (some CCL ops support `output_layout=TILE`), we skip the post-AG tilize entirely.

**Action — entry 5 (LM-head Untilize, 1309 μs):** the LM head output `1×1×129280×896` is BF16 TILE and gets Untilized for `from_device` / `argmax`. `ArgMaxDeviceOperation` accepts TILE input — verify and drop the Untilize.

### Class C — FP32 routing-weight scaffold survivor (entries 6, 9, 11)

`1×1×1×256` TilizeVal (2048 calls / 1274 μs/iter), `1×1×32×16384` UntilizeVal (2112 calls / 474 μs/iter), `1×1×32×256` UntilizeVal (2048 calls / 315 μs/iter). **All three are FP32 with extremely high call counts** (2048 = 32 devices × 64 sites/iter).

These survive *because* E38 replaced the score-computation but kept the **downstream** `matmul_29 (one-hot scatter) → reshape_141 → concat_22 → all_gather_24 → reshape_142 → typecast_95 → to_layout_259` chain in FP32. Every layout flip between `matmul_29` (FP32 dtype) and `typecast_95` (FP32→BF16) is an FP32 TM op.

**Action — fold dtypes:** flip `matmul_29` from `dtype=FP32` to `dtype=BFLOAT16` (E14-style cast-on-pack). The output is going to be BF16 anyway after `typecast_95`. Once `matmul_29` outputs BF16, the intermediate `reshape_141 → concat_22 → all_gather_24 → reshape_142` ops all become BF16, halving their per-element cost AND eliminating the explicit `typecast_95`. This is one of the cleanest wins on this list — same exploit as E14 and E17.

Expected saving from this single fold: 1274 + 474 + 315 = ~2063 μs/iter (if all three high-count sites are in the same FP32 chain and they all disappear). Likely ~1000 μs realistic (some FP32 ops will remain BF16, others will remain layout-flip).

### Class D — INT32 indexer-mask scaffold (entries 9, 15)

`1×1×32×16384` UntilizeVal FP32 (474) and `1×1×16384×32` UntilizeVal INT32 (183) — both happen inside the V3.2 indexer-mask chain (`scatter → reshape → mesh_partition → to_layout`) that we noted in `TILIZE_ANALYSIS.md` is now **structurally dead** (the indexer mask output `reshape_60`/`reshape_111` used to feed `add(typecast_55, reshape_60)` → `softmax_0`, but E41 collapsed `add+softmax+matmul_9` into the SDPA call which handles its own causal mask). The scatter writes to `var_71` have no other consumer (verified via `grep -n var_71 main.py` — only the indexer sites).

**Action — delete the V3.2 indexer-mask scaffold chain.** Both layer-0 (`main.py` ~5103–5267 in pre-E41 numbering) and layer-1 mirrors. Estimated saving: ~657 μs/iter just from entries 9 + 15, plus the FP32 binary / typecast / scatter ops embedded in the chain that aren't in the TM accounting.

### Class E — entries 7, 13 (small MLA-attention layout flips)

`1×16384×32×2` BF16 TilizeVal (1167) and `32×16×32×32` FP32 UntilizeVal (275) are inside the MLA Q/K production (the half-rotate RoPE in the `[..., 32, 2]` real/imag interleaved form). These won't disappear without restructuring the RoPE chain to use `ttnn.experimental.rotary_embedding_llama` — same conclusion as the earlier exit notes. **Park.**

## Implementation plan — what to actually try

Ordered by expected per-iter μs / risk:

1. **E42 (Class D): drop the V3.2 indexer-mask scaffold chain.** Two sites, both now dead after E41. Mechanical: delete the lines between the `var_71` scatter setup and `reshape_60`/`reshape_111`, plus verify `var_71` has no other reader. Estimated saving 657 μs + the FP32/INT32 ops embedded in the chain that aren't called out individually here (probably another 200–500 μs of `BinaryNg`, `Reshape`, `MeshPartition`, `Scatter`, `Slice` device ops). PCC risk low (we proved E41 doesn't need this mask).

2. **E43 (Class C): fold `matmul_29` and friends from FP32 to BF16.** Same pattern as E14. Single dtype switch + drop the now-unused `typecast_92`/`typecast_95` ops. Estimated saving 1000–2000 μs. PCC risk: should be small (bf16 → bf16 path is what `typecast_95` was producing anyway), but verify.

3. **E44 (Class B entry 5): drop the LM-head Untilize.** Check whether `ArgMaxDeviceOperation` accepts TILE. If yes, remove the explicit `to_layout(ROW_MAJOR)` before argmax. Estimated saving ~1300 μs. Low risk.

4. **E45 (Class B entry 3): fuse the hidden-state broadcast all_gather to output TILE.** Trickier — requires CCL config tweaks. Estimated 1600 μs.

5. **E46 (Class A audit): the weight tilize entries.** Need a one-shot diagnostic inside `_main` to verify whether they're real tilize calls or fast-path no-ops. If they're fast-path, no work. If they're real, fix the const_eval emitter to produce TILE.

If E42 + E43 + E44 land cleanly, expected total saving: **~3000–4000 μs / iter** (≈ 6–8 % of decode_1). Less dramatic than E38 / E41 individually, but additive on top of those.

## Open question worth a 5-min check

The top-1 Tilize entry `1×8×7168×2048 BF16` runs **4 times per iter per device** for **4902 μs total** (a per-call cost of ~1225 μs that doesn't quite add up — a single tilize at that shape should be much cheaper). Either (a) the kernel is real and the work is huge because the weight tensor really is being re-tilized every iteration, OR (b) the cost shown is mislabeled and is actually the time the host blocks on a fast-path no-op while the corresponding `SparseMatmulDeviceOperation` does its actual work. If (a), there's a const_eval bug to fix and 4902 μs is reclaimable. If (b), the perf accounting is just confusing and there's nothing to do. A `ttnn.get_tensor_layout()` check at every `sparse_matmul` weight input would resolve which.
