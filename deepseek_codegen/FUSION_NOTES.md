# DeepSeek-V3.2 codegen_py — fusion inventory

What was fused, what was tried and rejected, and what was never attempted. Reasons in the "wanted but couldn't" section are re-verified against tt-metal source on `third_party/tt-mlir/src/tt-mlir/third_party/tt-metal/src/tt-metal/` (commit hashes in `TUNING_LOG.md` and `TUNING_SUMMARY.md`).

> Convention: a "fusion" here means any change that collapses N device ops into fewer ops — kernel-level (e.g. `silu→multiply` via `input_tensor_a_activations`), op-level (e.g. swap a per-op chain for a single fused tt-metal kernel), or structural (e.g. stack two same-input matmuls).

---

## Landed fusions

Sorted by complexity — experimental tt-metal kernel swaps and structural rewrites at the top, single-kwarg algebraic folds at the bottom. "Complexity" here folds together: required understanding a new tt-metal kernel's API + constraints, lines of code added, number of sub-iterations to land, and whether it required new `main_const_eval_*` infrastructure.

| # | iter | what was fused | mechanism | landing commit |
| --- | --- | --- | --- | --- |
| 1 | **E47** | All 6 RoPE sites (4 binary FP32 mults + addcmuls + concat + strided slice + half-concat conversions per site) | `ttnn.experimental.rotary_embedding_llama` (prefill mode). New `trans_mat` const_eval (32×32 BF16 TILE half-rotate matrix matching `tt_transformers/tt/common.py::get_rot_transformation_mat`) + doubled cos/sin caches via `repeat_interleave(2,-1)`. **9 sub-iterations**, batch×n_heads collapse to seq_len trick. | `df948e60b` … `62355e583` |
| 2 | **E41** | 10-op MLA attention chain (`matmul_7 + matmul_8 → add → multiply(scale) → where(indexer-mask) → add(causal) → softmax → matmul_9`) in both layers | `ttnn.transformer.flash_multi_latent_attention_decode` (unpaged-causal MLA path, `V=None`, `head_dim_v=512`, custom `SDPAProgramConfig(grid=(8,8))`). Required unified Q `[1,32,16,576]` and K `[32,1,128,576]` build via `concat(dim=-1)`, internal-scale param `0.134765625`, debugging 3 prior cold-compile false-hangs. | `77cbff41b` |
| 3 | **E38** | The full per-op MoE router (sigmoid → +bias → group reshape → topk-2 per group → sum → topk-4 → mask → topk-8 → unbiased gather → normalize → ×scale) | `ttnn.experimental.deepseek_grouped_gate` (one fused kernel call). FP32/INT32 bridges to maintain downstream `matmul_29` + `all_gather_27` wiring; bias broadcast via `ttnn.repeat([1,1,32,1])` to satisfy the kernel's "bias shape == scores shape" validator. **~50 ops → 1 kernel call** (−461 LoC). | `2f88e489c` |
| 4 | **E30 + E31 + E32 + E33 + E34** | 2 same-input sparse_matmuls (gate `[7168×2048]` + up `[7168×2048]`) | **Structural fusion (not a kernel op):** new `main_const_eval_gate_up` pre-concatenates gate+up weights along N (on-device `ttnn.concat(dim=-1)` + host round-trip + BFP8 typecast), runs **one** wider `sparse_matmul [7168×4096]`, slices the output. Then 4-iteration tuning sweep on grid / `per_core_N` / `in0_block_w` (E31 stacked→(8,8), E32 down→(8,7), E33+E34 K-block widening). | `7433620b8`, `1585b446e`, `0d82e13a7`, `8132ca240`, `835b43a04` |
| 5 | **E6** | `to_layout(TILE)` immediately after `all_to_all_combine_0` | `ttnn.experimental.deepseek_moe_post_combine_tilize` (L1 ND-sharded TILE output, `nd_shard_spec(shard_shape=[32,3584], grid=8×8)`) + `ttnn.to_memory_config(... DRAM, INTERLEAVED)` bridge. Took **2 attempts** — E5 first tried `ttnn.sharded_to_interleaved` (legacy 1D path, doesn't handle ND shards, crashed `reduce_scatter`); E6 with `to_memory_config` worked. | `d34ac565c` |
| 6 | **E42 + E43** | The displaced attention compute (kept emitting after E41) — including the entire V3.2 indexer pipeline that lost its only consumer | AST-based iterative DCE pass: build def→inputs graph, seed live set with `_main` returns + `paged_update_cache` side-effect inputs, propagate liveness, handle tuple-destructuring assigns (`v_82, v_83 = ttnn.topk(...)` → call dead only if ALL outputs dead). Removed 86 dead `Call` statements / 839 LoC. | `7aa606a2e`, `bd79c83da` |
| 7 | **E1** | `pow → sum → reshape` (pre-allgather RMSNorm head) at 5 sites | `ttnn.rms_norm_pre_all_gather(dtype=FP32, HiFi4, fp32_dest_acc_en=True)` — partial fusion, post-allgather chain kept manual. Required FP32 precision preservation through the pre op + extra `slice([1,1,32,1])` to extract column 0 from the kernel's tile-wide stats. Sets up the (unsuccessful) E2/E7/E13 attack on the full post-allgather fusion. | `92c63cfea` |
| 8 | **E11** | `mul + addcmul` for the 4-binary-op RoPE pattern at 8 sites | `ttnn.addcmul` (ternary FMA). 32 binary ops → 16 ternary ops. Required use-after-free audit when moving deallocs of shared `x_second` and `typecast_45` operands across all 8 sites. **Superseded** by E47 swap to `rotary_embedding_llama`. | `951bec4b8` |
| 9 | **E8 + E9 + E10** | 130 sequential const-zero scatters writing into a shared `[16384]` buffer (layers 0+1) + 2 dynamic-source scatters | Semantic IR simplification — replaced with 1 scatter using the union index. Required proving provenance: 64 source slices `var_1..var_64` are all from one all-zero buffer (`main_const_eval_1`'s `zeros + all_gather + reshape`), indices are disjoint and cover the full buffer. Then 1 op = 64 ops with the union. | `64ed1d31e`, `6445c60d0`, `bce33b910` |
| 10 | **E24** | `silu → typecast → multiply` (SwiGLU, FP32, 2 MLA sites) | `input_tensor_a_activations=[SILU]` on the FP32 multiply — packs SiLU into the BinaryNg compute stage at the FP32 dtype. | `e53636d31` |
| 11 | **E23** | `silu → multiply` (SwiGLU, BF16 MoE expert FFN, 1 site) | `input_tensor_a_activations=[ttnn.UnaryOpType.SILU]` on the multiply. First real unary→binary kernel fusion landed. | `1610b963e` |
| 12 | **E25** | `matmul → relu → multiply` (2 sites) | Move ReLU from `matmul(activation="relu")` (no-op syntactic sugar on DRAM-interleaved matmul — confirmed via E12) into the downstream `multiply(input_tensor_a_activations=[RELU])`, where it actually packs into the BinaryNg kernel. | `28cd15478` |
| 13 | **E28** | `permute([0,2,1]) → matmul` at 6 MLA sites | `matmul(transpose_b=True)` — kernel does the transpose during the read of input B. Free fold. | `edf6cb33b` |
| 14 | **E46** | `permute([1,0,2,3]) → matmul → permute([2,0,1,3])` SDPA-output permute pair, both MLA layers | Algebraic fold: `permute([1,0,2,3]) ∘ permute([2,0,1,3]) = permute([2,1,0,3])`. | `d1ac5eb37` |
| 15 | **E14 + E17 + E27 + E45** | `multiply/add(...,dtype=FP32) → typecast(BF16)` and inverse widening | `dtype=` cast-on-pack on BinaryNg — the pack engine downcasts FP32→BF16 (or widens BF16→FP32) on the way out without a separate Typecast kernel. 11 sites total across 4 iterations (E14 multiply, E17 add, E27 widening, E45 routing chain). | `79a73ec0d`, `668c9e1a5`, `bdc9c56e8`, `e5af1111f` |
| 16 | **E19** | `scatter → to_layout(TILE) → reshape → to_layout(RM) → mesh_partition` (layout round-trip) | Skip layout flips entirely (ReshapeView accepts RM INT32). | `26c9c313b` |
| 17 | **E40** | 3 host round-trips (`from_device → to_layout/typecast → to_device`) in the MoE dispatch/remap/sparsity bridges | On-device `ttnn.to_layout` and `ttnn.typecast` — codegen's host trips were unnecessary conservatism. | (in `77cbff41b`) |

---

## Wanted but couldn't land (each reason re-verified)

### MoE compute family — all topology-incompatible with our 4×8 BH galaxy

| op | path | reason couldn't land | how I verified |
| --- | --- | --- | --- |
| `ttnn.experimental.moe_compute` | `experimental/ccl/moe_compute/` | Test infrastructure gates on `TT_MESH_GRAPH_DESC_PATH ∈ { single_galaxy_1x16_torus_graph_descriptor.textproto, single_galaxy_1x8_torus_graph_descriptor.textproto }`. The DeepSeek-V3 model entry in `MODELS_1x16` is parameterised at `mesh_shape=(1,16)` with `experts_per_device=2`. **Our setup is 4×8 BH with `experts_per_device=8` — neither mesh nor experts-per-device matches any tested config.** | `tests/nightly/tg/ccl/moe/test_moe_compute_6U.py:33-37` defines `MESH_GRAPH_DESC_{1x16,1x8}`; lines 1181-1184 + 1818 gate all tests on those descriptors. Default `experts_per_device_values=(2,)` (file:65). |
| `ttnn.experimental.moe_gpt` (sibling kernel in same dir) | `experimental/ccl/moe_compute/` | Same family, same gating. | (same source as above) |
| `ttnn.experimental.selective_reduce_combine` | `experimental/ccl/moe/selective_reduce_combine/` | Same `TT_MESH_GRAPH_DESC_PATH=1x8/1x16` gating in its tests. | `tests/nightly/tg/ccl/moe/test_selective_combine_6U.py:25,672,681`. |
| `ttnn.experimental.deepseek_moe_reduce_scatter` | `experimental/ccl/deepseek_moe_reduce_scatter/` | Kernel is registered for Wormhole only — both t3000 and tg test variants are decorated `@skip_for_blackhole("Requires wormhole_b0 to run")`. | `tests/nightly/t3000/ccl/test_deepseek_moe_reduce_scatter.py:145` and `tests/nightly/tg/ccl/test_deepseek_moe_reduce_scatter_6U.py:11`. |

### Deepseek-named single-op fusions — wormhole hardware assumptions

| op | path | reason couldn't land | how I verified |
| --- | --- | --- | --- |
| `ttnn.experimental.deepseek.moe.moe_gate_mm` | `experimental/deepseek/moe/moe_gate_mm/` | **Hard-fails on Blackhole:** `TT_FATAL(num_cores == 12, "moe_gate_mm requires exactly 12 DRAM-aligned cores (Wormhole); got {}. This op's ring algorithm is hardcoded for Wormhole's 12 DRAM views and does not support other architectures.")`. BH has a different DRAM-bank-to-core mapping. | `ttnn/cpp/ttnn/operations/experimental/deepseek/moe/moe_gate_mm/device/moe_gate_mm_program_factory.cpp:30-37` — exact message includes "does not support other architectures". |
| `ttnn.experimental.deepseek.mla.matmul_wo` | `experimental/deepseek/mla/matmul_wo/` | Kernel requires (a) input `Q` HEIGHT_SHARDED L1 with `shard_shape=(M, K)` row-major, (b) weight `W` HEIGHT_SHARDED DRAM with a replicate-across-DRAM-banks pattern, (c) **pre-allocated** output tensor with sharded `MemoryConfig`, (d) 7 collector cores carved out of the non-DRAM grid. Our codegen's `wo` is DRAM-INTERLEAVED with **row-parallel** weight (`wo.weight=(None,"model")` in the loader → K-split + reduce_scatter after). The kernel implements **column-parallel** wo. Drop-in is impossible — it's a loader+codegen rewrite, not a Python knob. | `device/matmul_wo_program_factory.cpp:25-65` does `find_collector_core_coords(.., 7)` + DRAM-bank shard reads; `device/matmul_wo_device_operation.cpp:14-32` validates "rank ≥ 2" but takes the output `MemoryConfig` from the caller's pre-allocated tensor. `tests/ttnn/nightly/unit_tests/operations/experimental/test_mla_wo.py:206-232` shows the required `in0_shard_spec` HEIGHT_SHARDED L1 + DRAM-banked W. |

### MLA QKV-pack fusions — graph shape doesn't admit

| op | path | reason couldn't land | how I verified |
| --- | --- | --- | --- |
| `ttnn.experimental.create_qkv_heads_from_separate_tensors` | `experimental/transformer/create_qkv_heads_from_separate_tensors/` | Validator hard-requires both Q and KV inputs **BLOCK_SHARDED L1** ("Q input tensor memory layout must be BLOCK_SHARDED but got {}"). Also requires KV packed as K\|V on the last dim and `q_shape[0]==kv_shape[0]`. **DeepSeek MLA produces K (from `matmul_6`, on the rms-normed kv-latent) and V (from `slice_68(matmul_0) → all_gather → sum → slice → rms_norm_1`) on completely separate upstream chains — they are never co-resident in a single KV tensor.** Adding an L1 BLOCK_SHARDED bridge also forces `batch == num_h_cores` grid commitment. | `device/create_qkv_heads_from_separate_tensors_device_operation.cpp:35` ("Operands to TM must be sharded"); lines 44 + 48 ("must be BLOCK_SHARDED"). |
| `ttnn.experimental.nlp_create_qkv_heads_decode` | `experimental/transformer/nlp_create_qkv_heads_decode/` | Even stricter: takes a **single** QKV-packed input. Our Q comes from `matmul_2` (on q-latent) and K from `matmul_6` (on kv-latent) — different upstream matmuls on different inputs. | TUNING_LOG.md "Exit notes (after MLA-QKV survey)" section. |

### RMSNorm post-allgather — internal tt-metal bug

| op | path | reason couldn't land | how I verified |
| --- | --- | --- | --- |
| `ttnn.rms_norm_post_all_gather` (E7) | `normalization/rmsnorm_distributed/` | Aborts with `TT_FATAL @ slice.cpp:35: input_rank == begins.size() -- Input rank 4 and begins 2 must have the same size`. The slice is inside the op's internal stats-extraction path (no Python frame in the trace beyond `rms_norm_post_all_gather`). Our gathered stats shape `[1,1,32,256]` (rank 4) trips the kernel's hardcoded rank-2 `begins=[0,0]`. | `ttnn/cpp/ttnn/operations/data_movement/slice/slice.cpp:35` has the TT_FATAL with **exactly** the message above. Dispatch chain: `ttnn::rms_norm_post_all_gather` → `ttnn::prim::layer_norm_post_all_gather` (file `rmsnorm_post_all_gather.cpp:42-55`) → internal `slice(stats, ...)` with rank-2 begins. |
| `ttnn.experimental.wan_fused_rmsnorm_post_allgather` (E13) | `experimental/transformer/fused_distributed_rmsnorm/` | WAN variant **inherits the same internal slice kernel** that hard-codes `begins.size()==2`, so a rank-4 `[1,1,S,H]` input triggers the same assertion regardless of `num_heads_per_device`. | Same `slice.cpp:35` TT_FATAL; verified by attempting the swap at one site (E13 attempt) and observing identical crash signature. |
| Full `rms_norm_post_all_gather` fusion with BF16 stats + `all_gather(dim=3)` (E2) | `normalization/rmsnorm_distributed/` | Validation **passes** (call shape matches the upstream reference test) but kernel produces NaN/Inf — KV cache `max\|Δ\|` reaches 2.2e38, sampled tokens diverge 100 %. Golden-PCC reads 1.0 *false positive* because `inf/inf` collapses in the correlation calc. Hypothesised cause (per TUNING_LOG): the pre op's tile-wide stats output has undefined columns 1-31 per device; the post stage's row-reduce averages garbage. | Empirical, not source-verified. Documented in TUNING_LOG.md row E2. Retry would need a `ttnn.to_torch` dump at one site to confirm root cause before attempting at 5 sites. |

### Sparse_matmul tuning knob

| knob | reason couldn't land | how I verified |
| --- | --- | --- |
| `out_subblock_w > 1` on `ttnn.sparse_matmul` (E36, E37) | Kernel hangs the device. Two consecutive hangs at `out_subblock_w=2` on both `per_core_N=4` (E36, both matmuls) AND `per_core_N=2` (E37, stacked-only, exactly 1 sub-block per core). Each required `kill -KILL` + `tt-smi -glx_reset_auto`. Falsifies the hypothesis that multi-sub-block reduction is the trigger — even single-sub-block multi-tile-wide hangs. Conclusion: the sparse-mask reader path assumes a 1-tile-wide pack pattern. **The `tt-perf-report`'s "Output subblock ≥ 2" hint is a generic matmul hint and does NOT apply to `ttnn.sparse_matmul`.** | Empirical, not source-documented. Reproduced twice. Park until a tt-metal fix lands. |

### Single-op fusions where the graph pattern is absent

| op | path | reason couldn't land |
| --- | --- | --- |
| `ttnn.experimental.dit_rms_norm_unary_fused` | `experimental/transformer/dit_rms_norm_unary_fused/` | Requires a `rms_norm → silu/gelu` adjacency on the same buffer. Our silu sites are downstream of `all_gather` / `reshape`, never directly downstream of an `rms_norm`. |
| `ttnn.experimental.dit_minimal_matmul_addcmul_fused` | `experimental/transformer/dit_minimal_matmul_addcmul_fused/` | `matmul + addcmul(x, y, scalar)` pattern not present in the DeepSeek decode graph. |
| `ttnn.experimental.deepseek_prefill.{combine,dispatch,extract,insert,masked_bincount,moe_grouped_topk,offset_cumsum,post_combine_reduce,routed_expert_ffn}` | `experimental/deepseek_prefill/*` | Designed against prefill batch shapes (long `seq_len`, dense expert dispatch). Our graph is decode-only — every prefill kernel either crashes on a shape constraint or takes a slow fallback worse than the current decode ops. |

### CCL fusion attempts that landed but regressed

| iter | what | reason kept out | how I verified |
| --- | --- | --- | --- |
| **E48 (a) `num_links=2` globally** | per-link parallelism | PCC PASS but +61 μs / +0.27 % regression — per-link overhead exceeds throughput gain at our op sizes. | Empirical. |
| **E48 (b) `num_links=4`** | push beyond 2 links | TT_FATAL: "Requested link index 2 is out of bounds. 2 ethernet channels available" — BH-galaxy adjacency has exactly 2 ethernet channels. | Empirical (kernel-side enforcement). |
| **E48 (c) `Topology.Linear` globally** | swap Ring → Linear | PCC PASS but +2,144 μs / +9.5 % regression — Linear is far slower than Ring on our patterns. AllGather alone rose 3,263 → 5,080 μs. | Empirical. |
| **E48 (d) `ttnn.experimental.all_reduce_async` for one `reduce_scatter + all_gather` pair** | fuse the targeted RS+AG | PCC bit-identical to E47. Targeted pair dropped −1,413 μs, but other AllGathers slowed 1,505 → 2,690 μs → **net +147 μs.** Hypothesis: persistent semaphores on cores `(0,0)-(7,7)` interfere with other CCL ops' worker-core placement. | Empirical. |

### Smaller fusions deemed not worth the wiring

| op | path | reason couldn't justify the cost |
| --- | --- | --- |
| `ttnn.experimental.all_gather_concat_heads_fused` | `experimental/ccl/all_gather_concat_heads_fused/` | Needs persistent semaphore + sub_device id. The head-merge is cheap; total saving across MLA sites is small. ROI dominated by wiring cost. |
| `ttnn.experimental.paged_cache.fused_update_cache` | `experimental/paged_cache/device/fused_update_cache/` | `paged_update_cache` itself is only 32 μs / 6 ops. The surrounding layout bridges that *could* fold in are also small. Optimistic ceiling 100-300 μs total — below the threshold for the wiring it'd add. |
| `ttnn.experimental.llama_reduce_scatter_matmul` / `matmul_reduce_scatter_async` / `llama_all_gather_matmul_async` / `minimal_matmul_strided_reduce_scatter_async` / `strided_all_gather_minimal_matmul_async` | `experimental/ccl/*` | Each requires a clean `matmul → reduce_scatter` (or `all_gather → matmul`) adjacency with no intervening op. Our matmuls almost all have intervening reshape/typecast/silu — clean adjacencies are rare. Per-pair saving ~50-200 μs, finding pairs needs a per-op trace. |
| `ttnn.experimental.deepseek_moe_fast_reduce_nc` (manual binding) | `experimental/reduction/deepseek_moe_fast_reduce_nc/` | Already auto-dispatched by `ttnn.sum` on N/C reductions (we have 6 such ops). Remaining 8 generic `ReduceDeviceOperation` sites total 40 μs — not worth a manual swap. |
| `ttnn.moe_routing_remap` | `data_movement/moe_routing_remap/` | Our `moe_expert_token_remap` (170 μs / 1 op) is already the kernel-level remap; this op is a different/older shape and doesn't replace it cleanly. |
| `ttnn.square(x)` (vs `pow(x, 2)`) | `eltwise/unary/square` | Already structurally removed in E1 via the pre_all_gather fusion (the `pow` was inside the manual chain that E1 replaced). |
| `activation=` kwarg on `ttnn.matmul` (DRAM-interleaved) | `eltwise/matmul/` | Confirmed via E12 to be **no-op syntactic sugar** on DRAM-interleaved BF16/BFP8 matmul — TTNN auto-launches a follow-up Unary kernel instead of packing the activation into the matmul kernel. Real activation fusion needs sharded matmul with `MatmulMultiCoreReuseProgramConfig` and `fused_activation=` in the `program_config`. Worked around by pushing the activation into the downstream BinaryNg multiply (E25). |

---

## Never tried (deferred to a separate project)

These are documented in plan files and parked because each requires a multi-iteration structural rewrite, not a single-commit patch.

| target | what would unblock it | plan doc |
| --- | --- | --- |
| **L1-sharded `sparse_matmul` input** (the surviving `SLOW` / "place in L1" hint on the 2 surviving sparse_matmuls) | Pre-sharded upstream tilize to L1, `memory_config=L1_*` on the sparse_matmul, sharded→interleaved bridge before any DRAM-input downstream op. Input A is `[32, 7168]` BF16 ≈ 458 KB — comfortable for L1. | TM_REDUCTION_PLAN.md |
| **`flash_multi_latent_attention_decode` paged-cache reformat** | Combine the separate `compressed_kv` and `k_pe_cache` into one packed `kvpe_cache` at cache-write time (per layer). Unlocks the kernel's HEIGHT_SHARDED L1 fast path that the demo's `mla1d.py` uses. Currently using the unpaged-causal MLA path with on-the-fly `concat`. | (called out in `FLASH_MLA_HANG_REPRO.md` + TUNING_LOG E41 notes) |
| **`matmul_wo` column-parallel flip** (enables `ttnn.experimental.deepseek.mla.matmul_wo`) | Loader change `wo.weight = ("model", None)` (CP weight split) instead of `(None, "model")` (RP) in tt-forge-models, then re-emit the codegen. Removes the post-wo `reduce_scatter`. | TUNING_LOG.md post-E38 op survey |
| **Async CCL with cluster-wide persistent semaphore pool** | Extend the E48(d) `all_reduce_async` fusion to **all** cluster_axis=1 RS+AG pairs in one iteration so the worker-core conflict averages out across the graph. E48(d) only swapped one pair and ate the regression on the others. | TUNING_LOG E48 row |
| **`all_to_all_async_generic` / `all_gather_async`** | Persistent `GlobalSemaphore` plumbing in const_eval. Async overlap of comm with compute. Marginal at our op-to-op gap pattern. | — |

---

## Quick cross-reference

Counts at a glance:

- **Landed fusions:** 17 iterations across kernel-fusion (rms_norm_pre, deepseek_grouped_gate, flash_multi_latent_attention_decode, rotary_embedding_llama, deepseek_moe_post_combine_tilize), kernel-level activation fusion (silu/relu/transpose_b/cast-on-pack), structural fusion (stacked sparse_matmul, scatter coalesce), and post-fusion DCE.
- **Tried and rejected with verified reason:** 16 ops — 5 topology-incompatible (4×8 BH vs WH-TG-only), 2 with tt-metal internal bugs (slice.cpp:35), 1 with no fix and hangs the device (sparse_matmul `out_subblock_w>1`), 4 where the graph pattern is absent (dit_*, prefill_*), 4 CCL knob attempts that regressed elsewhere or hit hardware limits.
- **Not even tried (deferred):** 5 — all multi-iteration structural rewrites.

The honest summary: the high-ROI fusion lane on this graph is **wholly exhausted** at the single-commit cadence. Every additional %  requires a structural rewrite of either the codegen emitter, the tt-forge-models loader, or tt-metal itself.
