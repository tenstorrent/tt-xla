# DeepSeek-V3.2 codegen_py decode tuning — replay guide

If you started from `graph_0/main.py` (parity-restored E0 baseline, 319,925 μs / step) and wanted to land the same ≈92 % decode-DT reduction with no dead ends, this is the actual ordered list. Each row links the landing commit on `mvasiljevic/deepseek-router-fuse`. Per-step decode after the final kept iteration: **≈22,523 μs** (vs baseline ≈298,000 μs apples-to-apples per-step inside the warm loop).

Full history with revert iterations and per-op tracy deltas lives in `TUNING_LOG.md`. This file is the curated "actual recipe."

## Cumulative milestones

| stage | decode DT (μs) | Δ vs baseline | landing commit |
| --- | ---: | ---: | --- |
| **E0** baseline (parity-restored + golden-PCC harness) | 319,925 | — | `632a35d56` |
| after scatter coalesce (E8+E9) | 309,880 | −3.2 % | `6445c60d0` |
| after stacked sparse_matmul fusion (E30) | 238,025 | −25.6 % | `7433620b8` |
| after sparse_matmul grid/block sweep (E31–E34) | 80,602 | −74.8 % | `835b43a04` |
| after router fusion (E38) | 54,665 | −82.9 % | `2f88e489c` |
| after SDPA fusion (E41) + dead-code elimination (E42/E43) | 24,073 | ≈−92 % | `bd79c83da` |
| after RoPE kernel swap (E47, per-step) | 22,523 | ≈−92.4 % | `62355e583` |

> Five iterations (E30, E31, E32, E33, E38) together account for **≈93 %** of the total saving. Everything else is rounding.

---

## The 5 must-do iterations (biggest leverage)

### 1. Stacked gate+up sparse_matmul fusion — **−71,559 μs** (commit `7433620b8`, E30)

The MoE FFN has two same-input matmuls: `b={128} × 32 × 7168 × 2048` for gate and the identical shape for up. Both consume the same input + sparsity. Pre-concatenate `[gate; up]` along N into one `[1, 8, 7168, 4096]` BFP8_B TILE weight (a new `main_const_eval_gate_up`), call **one** `ttnn.sparse_matmul`, then `ttnn.slice` the output into the two halves before the SwiGLU multiply.

Key calibration: `tt-perf-report`'s `SLOW / 2 GB/s` hint is **per-core, not aggregate**. The original kernel was core-bound at 11/110 cores; doubling N to 4096 while keeping `per_core_N=6` brings 21 cores online and the new wider matmul (107 ms) runs *less* time than the two narrow ones it replaced (89 ms × 2 = 178 ms).

### 2. Size the matmul grid to the work — **−116k μs cumulative** (commits `1585b446e`, `0d82e13a7`, E31+E32)

For both surviving sparse_matmuls, drop the default `(11,10)` / `(12,10)` device-max grids and pick a sub-grid that **exactly tiles the N-stripe count**:

- Stacked gate+up: grid `(8,8) = 64 cores` with `per_core_N=2` → N=128 tiles / 2 = 64 stripes, one per core. Drops 107,961 μs → 48,782 μs (−59,178 μs, **2.2× speedup, 4.9 TFLOPs**).
- Down: grid `(8,7) = 56 cores` with `per_core_N=4` → N=224 tiles / 4 = 56 stripes. Drops 83,399 μs → 26,657 μs (−56,740 μs, **3.1× speedup, 4.5 TFLOPs**).

Lesson: **size the compute grid to the work, not to the device max.** 64 active cores beats 110 cores when only 11 of the 110 actually run.

### 3. K-direction block widening — **−34,165 μs** (commit `8132ca240`, E33; then `835b43a04`, E34)

Once the grid is sized, the next lever is `in0_block_w` (K-direction reader block). Set `in0_block_w=2` on both sparse_matmuls (E33: −34,165 μs / 45 % reduction on the SparseMatmul op category). Then push the stacked matmul to `in0_block_w=4` (E34: another −7,287 μs). Don't go further — E35 (`in0_block_w=8`) flatlined because per-core DRAM BW had stopped rising at 13 GB/s.

Lesson: **K-block widening wins until per-core DRAM bandwidth stops rising.** Measure per-matmul and stop when one side stops responding. Smaller-K matmuls (down, K=64 tiles) hit the ceiling earlier than larger-K ones (stacked, K=224 tiles).

### 4. Router block → `ttnn.experimental.deepseek_grouped_gate` — **−25,935 μs** (commit `2f88e489c`, E38)

Replace the ≈50-op manual router (`sigmoid → +bias → group reshape → topk-2 per group → sum → topk-4 of group sums → mask → topk-8 over masked → gather unbiased sigmoid scores → normalize → ×route_scale`) with one fused kernel call. The kernel takes pre-sigmoid logits + bias and returns `(scaled_normalized_weights, topk_indices)`.

The direct saving from removing router ops is small (≈500-1000 μs). The dominant **second-order** effect: the kernel produces *cleanly-zeroed* weights at non-selected experts, so `moe_expert_token_remap → SparseMatmul` sees only the strictly necessary 8-experts-per-token sparsity (vs ≈3.5× as many "active" expert-token pairs with the per-op chain's bf16 typecast / divide noise). SparseMatmul drops **33,901 → 9,592 μs (−71.7 %)** on this single change.

Wiring detail: BF16 input + bias (use `ttnn.repeat` to broadcast bias to scores shape — the kernel's validator requires identical shapes), then re-emit FP32/INT32 bridges so downstream `matmul_29` (one-hot scatter) and `all_gather_27` (indices broadcast) wire up unchanged. main.py shrinks ≈461 LoC.

PCC drops slightly (0.921 → 0.920) because the kernel uses bitonic-sort tie-breaking vs the per-op chain's generic topk — both are valid under the kernel's own `assert_in_valid_outcomes` tie-validator.

### 5. MLA attention → `flash_multi_latent_attention_decode` + dead-code elimination — **−9,623 μs combined** (commits `77cbff41b`, `7aa606a2e`, `bd79c83da`, E41/E42/E43)

The 2 MLA decoder layers each ran a 10-op attention chain (`matmul_7+8 → add → multiply(scale) → where(indexer-mask) → add(causal-mask) → softmax → matmul_9`). Replace with one `ttnn.transformer.flash_multi_latent_attention_decode` call per layer:

- Unified Q: `Q [1,32,16,576] = concat([Q_nope_absorbed [32,16,512], Q_rope [32,16,64]], dim=-1)`.
- Unified K: `K [32,1,128,576] = concat([compressed_kv [32,128,512], k_pe_cache [32,128,64]], dim=-1)`. Pass `V=None`, kernel reuses K's first 512 dims (`head_dim_v=512`).
- `scale=0.134765625` (the DeepSeek V3.2 MLA mscale; the chain applied it as a separate multiply between matmul and softmax — the kernel applies it internally).
- `SDPAProgramConfig(grid=(8,8), q_chunk_size=0, k_chunk_size=128)`.

E41 (the fusion) saves −4,155 μs directly. **Critical follow-ups:**

- **E42 (`7aa606a2e`)** — delete the now-dead `matmul_7/8 → add → multiply → where → typecast → add → softmax → deallocate(softmax)` ranges in both layers (≈234 lines each). The codegen emitter doesn't auto-DCE after a chain fusion. **−3,701 μs.**
- **E43 (`bd79c83da`)** — transitive DCE of the V3.2 indexer compute pipeline (Q-side projection, QK^T, softmax, topk, mask construction) — all orphaned because their only consumer was the deleted `where(indexer-mask)`. AST-based iterative DCE found 86 dead `Call` statements (32 reshape + 14 slice + 11 typecast + 10 multiply + 7 matmul + 7 concat + 7 all_gather + …). main.py shrinks ≈839 LoC. **−1,767 μs.**

Lesson, broad: **every chain-fusion landing should run iterative DCE in the same patch.** Without DCE, a displaced producer chain runs as garbage for thousands of μs per step.

Pitfall during debugging: `flash_multi_latent_attention_decode` takes **≥320 s to cold-compile** on first call. Our initial `timeout 180` reported "HANG" three times (E39, E40 v1/v2) — false positives. The kernel itself works on Blackhole with our exact shapes. See `FLASH_MLA_HANG_REPRO.md`. **Any future kernel-introduction iteration should use `timeout ≥ 600 s`.**

---

## The smaller-but-clean wins (do these too)

| # | iter | what | Δ DT | commit |
| --- | --- | --- | ---: | --- |
| 6 | **E8 + E9** | coalesce 128 sequential const-zero scatters in layers 0+1 into 1 scatter each (each writes disjoint 256-element index ranges → one scatter with the union index does the same work). Cost paid in `main_const_eval_1` which is free. | **−9,693 μs** | `64ed1d31e`, `6445c60d0` |
| 7 | **E47** | replace all 6 RoPE sites with `ttnn.experimental.rotary_embedding_llama` (prefill mode). Build cos/sin once per step from `freqs_cis [16384,32,2]` via `repeat_interleave(2,-1)`, add a `trans_mat` const_eval (32×32 BF16 TILE half-rotate matrix matching `models/tt_transformers/tt/common.py::get_rot_transformation_mat` — `M[2k, 2k+1]=+1, M[2k+1, 2k]=-1`). | **−1,513 μs** | `df948e60b` … `62355e583` (9 iterations) |
| 8 | **E45** | fold the routing-weight MoE chain to BF16 — change `matmul_29.dtype=BFLOAT16` (HiFi2), drop the upfront `typecast(BF16→FP32)` and the closing `typecast_95(FP32→BF16)` before `moe_expert_token_remap`. Adds one small BF16→FP32 typecast feeding `matmul_30`. | −35 μs | `e5af1111f` |
| 9 | **E28** | fold `permute([0,2,1]) → matmul(transpose_b=False)` into `matmul(transpose_b=True)` at 6 MLA matmul sites. Free fusion — the matmul does the transpose during input-B read. | −33 μs | `edf6cb33b` |
| 10 | **E25** | move `relu` from `matmul(activation="relu")` (no-op syntactic sugar on DRAM-interleaved) into `multiply(input_tensor_a_activations=[RELU])` (real fusion). 2 sites. | −36 μs | `28cd15478` |
| 11 | **E24** | fuse `silu→typecast→multiply` (FP32 SwiGLU) via `input_tensor_a_activations=[SILU]` on the multiply at 2 MLA sites. | −86 μs | `e53636d31` |
| 12 | **E23** | fuse `silu→multiply` (SwiGLU) on the MoE expert FFN. | −16 μs | `1610b963e` |
| 13 | **E14** | fold `multiply(γ_FP32, x_normalized_FP32, dtype=FP32) → typecast(BF16)` into `multiply(..., dtype=BFLOAT16)` at 6 sites — the BinaryNg packer can downcast FP32→BF16 on the way out without a separate Typecast kernel. | −10 μs | `79a73ec0d` |
| 14 | **E17** | same dtype= cast-on-pack trick on `add` at 2 attention-mask sites (`add → softmax` already wants BF16). | −43 μs | `668c9e1a5` |
| 15 | **E27** | dtype= cast-on-pack trick widening BF16→FP32 (and INT32→FP32) at 2 add sites — confirms the lever works in both directions. | structural | `bdc9c56e8` |
| 16 | **E19** | skip TILE round-trip around `scatter→reshape→mesh_partition` at 2 sites (ReshapeView accepts ROW_MAJOR INT32). | −39 μs | `26c9c313b` |
| 17 | **E11** | RoPE FMA fusion via `ttnn.addcmul` (8 attention sites, 32 binary muls → 16 ternary addcmuls). | −5 μs (structural cleanup, mostly superseded by E47) | `951bec4b8` |
| 18 | **E10** | coalesce the last two dynamic-source scatters into one (the 64-coalesce pattern from E8/E9, on a different chain). | −23 μs | `bce33b910` |
| 19 | **E6** | `ttnn.experimental.deepseek_moe_post_combine_tilize` for the `to_layout(TILE)` immediately after `all_to_all_combine_0`. Use `ttnn.to_memory_config(... DRAM, INTERLEAVED)` to bridge the L1 ND-sharded output back to interleaved before `reduce_scatter_10` (NOT `ttnn.sharded_to_interleaved` — that's the legacy 1D path and doesn't handle ND shards). | −189 μs | `d34ac565c` |
| 20 | **E3** | drop `MathFidelity.HiFi4 → HiFi2` on 30 BF16-output matmuls (kept HiFi4 for the 5 FP32-output matmuls including LM head). DRAM-bandwidth-bound matmuls don't care, and it's a free precision drop. | −171 μs | `eb8307269` |
| 21 | **E1** | strict-precision distributed RMSNorm pre-allgather fusion — replace `pow → sum → reshape` with `reshape → rms_norm_pre_all_gather(FP32, HiFi4, fp32_dest_acc_en=True) → slice → reshape`. Net neutral but sets up E2-style fusion attempts and removes structural redundancy. | neutral | `92c63cfea` |
| 22 | **E40** | collapse 3 MoE host round-trips at the dispatch/remap/sparsity bridges — replace each `from_device → to_layout(ROW_MAJOR)/typecast → to_device` triple with a single on-device call. | −65 μs (host gap not captured in tracy decode scope, real host saving) | (in `77cbff41b`) |
| 23 | **E46** | fold the SDPA-output permute pair in both MLA layers — `permute([1,0,2,3]) ∘ permute([2,0,1,3]) = permute([2,1,0,3])`. | −2 μs (cosmetic) | `d1ac5eb37` |

---

## The hard-won lessons (what NOT to try without protection)

These are the dead ends. Skipping them saves multiple cycles of `tt-smi -glx_reset_auto` per iteration.

| trap | symptom | iter(s) | mitigation |
| --- | --- | --- | --- |
| **`out_subblock_w > 1` on `ttnn.sparse_matmul`** | Hard kernel hang (no Python frame, 746 % CPU). Two confirmed hangs at `per_core_N=4` AND `per_core_N=2`. | E36, E37 | Park indefinitely. Sparse-mask reader path assumes 1-tile-wide pack. Needs tt-metal fix, not a Python knob. |
| **`ttnn.rms_norm_post_all_gather` / `wan_fused_rmsnorm_post_allgather` on rank-4 stats** | `TT_FATAL @ slice.cpp:35: input_rank == begins.size() -- Input rank 4 and begins 2 must have the same size`. | E7, E13 | Op's internal slice hardcodes `begins.size()==2`. Need either a metal-side fix or rank-2 stats reshape before the call (op also internally slices stats by hidden_size, so reshape alone won't fix). |
| **Full `rms_norm_post_all_gather` fusion (BF16 stats, dim=3 all_gather)** | Validation passes but kernel produces NaN/Inf (`max\|Δ\|` ≈ 2e38). Golden PCC reads 1.0 *false positive* because `inf/inf` collapses to 1 in the correlation calc. | E2 | The kernel's tile-wide stats output has undefined columns 1-31; row-reduce in the post stage averages garbage. Pre-flight any retry with a `ttnn.to_torch` dump at one site. |
| **`ttnn.experimental.moe_compute` / `moe_gpt` / `selective_reduce_combine` on BH 4×8** | Topology incompatible — tests hardcoded to `mesh_shape=(1,8)/(1,16)/(16,8)` with `experts_per_device∈{1,2}`. We have 4×8 with `experts_per_device=8`. | (surveyed) | Park until tt-metal lands a BH 4×8 program factory. |
| **`ttnn.experimental.deepseek_moe_reduce_scatter`** | `@skip_for_blackhole("Requires wormhole_b0 to run")` — kernel only registers a WH program factory. | (surveyed) | Park. |
| **`ttnn.experimental.deepseek.moe.moe_gate_mm`** | `TT_FATAL(num_cores == 12)` — hardcoded to WH's 12-DRAM-core ring. | (surveyed) | Park. |
| **`ttnn.experimental.create_qkv_heads_from_separate_tensors` / `nlp_create_qkv_heads_decode`** | Requires Q+K+V packed on the last dim, both TILE+BLOCK_SHARDED, batch==num_h_cores. DeepSeek MLA produces K and V on separate chains from different upstream matmuls — they're never co-resident in one tensor. | (surveyed) | Park. The graph shape doesn't admit this fusion without re-emitting the upstream chain. |
| **`activation=` kwarg on `ttnn.matmul` (DRAM-interleaved)** | UnaryDeviceOperation count doesn't drop; the activation auto-launches as a follow-up Unary kernel — syntactic sugar only. | E12 | Real activation fusion requires sharded matmul with `MatmulMultiCoreReuseProgramConfig` and `fused_activation=` in the program_config. Or push the unary into the *downstream* `multiply`'s `input_tensor_a_activations` (E25). |
| **`addcmul` on tiny `[32,1]` or INT32 tensors** | Per-op kernel-launch overhead dominates. Regresses vs `mul + add`. | E15 (RMS stats), E18 (INT32) | Prefer addcmul only when tensors are FP32/BF16 AND ≥ `[32, 32]` so math density amortises the launch cost. |
| **`out_subblock_h=1, out_subblock_w=2` reshape fold into L1 sharded** | Reshape kernel's internal sharded-write path is slower than separate reshape (DRAM) + to_memory_config (to L1). | E29 | Layout-fold isn't always a win — verify the producer's "sharded output" code path is at least as fast as the separate-kernel default. |
| **Removing layout flips around `mesh_partition` on small/non-tile-aligned shapes** | `mesh_partition` on ROW_MAJOR INT32 with non-tile-aligned last dim is slower than the two layout conversions + TILE mesh_partition. | E20, E21 | Verify the new ROW_MAJOR consumer's output is also tile-row-major-friendly before removing flips. |
| **TM reorder around `sparse_matmul` output (Untilize before Permute+Reshape)** | RM-permute on dim-0/1 swap of 4D tensors is *more* expensive than TILE permute; Untilize on `[16,8,32,7168]` costs ≈2× Untilize on the post-reshape `[8,1,512,7168]`. | E44 | TILE↔RM cost depends strongly on the **shape** of data being converted, not just volume. The existing emit order was already at a local minimum. |
| **`flash_multi_latent_attention_decode` "hang"** | Zero stdout for >3 minutes under `timeout 180`. | E39, also misdiagnosed in early E40 attempts | Kernel cold-compile takes ≥320 s. **Use `timeout ≥ 600 s` for any iteration introducing a new fused kernel.** The unpaged-causal MLA path works fine on Blackhole with our shapes. |
| **CCL acceleration knobs (`num_links=2/4`, `Topology.Linear`, `all_reduce_async`)** | (a) `num_links=2` regresses +61 μs. (b) `num_links=4` crashes — BH-galaxy has exactly 2 ethernet channels. (c) `Topology.Linear` regresses +2,144 μs. (d) `all_reduce_async` saves 1.4 ms on the targeted pair but persistent semaphores shift other CCL ops' worker-core placement, net +147 μs. | E48 (4 sub-attempts) | CCL on this branch is at near-optimal default. Real gains require either fusing across **all** cluster_axis=1 RS+AG pairs (so worker-core conflicts average out), or `llama_reduce_scatter_matmul` / `matmul_reduce_scatter_async` (absorb the surrounding matmul into CCL), or sharded CCL I/O. Multi-iteration projects. |

---

## The harness (do this first)

Before any iteration, set up the PCC harness so you can tell which patches drift unsafely:

- **Bootstrap PCC (soft).** `pcc.py` runs `_main` once and compares each of the 10 returned tensors against `baseline_outputs.pt`. PCC=1.0 means bit-identical — catches refactor drift but is **not** a fidelity check.
- **Golden PCC (hard floor 0.9).** `_main` was patched to additionally return the pre-argmax logits tensor (`ttnn_to_layout_267`) as `out[9]`. Compare `chip[0].row[0]` of TTNN's `(32 chips, 32 batch_pad, 129280 vocab)` output against `golden_logits.pt[0]`. Baseline reads **0.921842** — that's the floor (BF16 quantization through LM head + reduce_scatter). Hard floor 0.9 = "as good as baseline."

> Pitfall: when a fused kernel produces NaN/Inf, PCC can read 1.0 as a **false positive** because `inf/inf` collapses to 1 in the correlation calc. Always sanity-check KV cache `max|Δ|` and at least a few logit values for finiteness when a "win" looks suspicious. (See E2.)

Landing the harness: commit `632a35d56` (E0 baseline + golden-PCC harness).

---

## Why stop at E47

After E47 the remaining levers are all structural multi-day projects, documented in their own plan files:

- `flash_mla_decode` paged-cache reformat (combined `kvpe_cache` instead of separate `compressed_kv` + `k_pe_cache`) — would unlock the kernel's height-sharded fast path. Currently using unpaged-causal.
- `matmul_wo` column-parallel flip (loader change `wo.weight = ("model", None)` instead of `(None, "model")`) — enables `ttnn.experimental.deepseek.mla.matmul_wo`. Saves ≈50 μs/site after re-emit.
- Async CCL with persistent `GlobalSemaphore` pool covering ALL cluster_axis=1 RS+AG pairs in one iteration so worker-core conflicts average out (E48 attempted only one pair, regressed elsewhere).
- L1-sharded sparse_matmul input prototype — the surviving `SLOW`/2 GB/s hint. Input A is `[32, 7168]` BF16 ≈ 458 KB / device, comfortable for L1, but requires pre-sharded upstream tilize.

None of these admit a single-line patch. The per-experiment loop on this branch is genuinely converged at ≈92 % reduction; remaining gains need a different cadence.

---

## TL;DR — the 5-step minimal replay

1. Land the PCC harness (`632a35d56`).
2. Coalesce const-zero scatter chains in both layer-0 and layer-1 (`64ed1d31e`, `6445c60d0`). **−3.2 %**
3. Stack gate+up into one sparse_matmul, then sweep `grid + per_core_N + in0_block_w` until each matmul saturates (`7433620b8`, `1585b446e`, `0d82e13a7`, `8132ca240`, `835b43a04`). **−71.6 % more (cumulative −74.8 %)**
4. Swap the router block for `ttnn.experimental.deepseek_grouped_gate` (`2f88e489c`). **−25.4 % more (cumulative −82.9 %)**
5. Swap MLA attention for `flash_multi_latent_attention_decode`, then DCE the displaced chains (`77cbff41b`, `7aa606a2e`, `bd79c83da`). **−45 % on the remaining DT (cumulative ≈−92 %)**

Everything else combined moved DT by less than 1.5 %.
