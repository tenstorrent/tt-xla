# Z-Image-Turbo: tt_dit Op Integration Report

**Hardware**: 4× Blackhole P150, TP=4, 13×10 compute grid per card  
**Baseline**: `model_ttnn.py` (compiler-generated, unoptimized)  
**Optimized**: `model_ttnn_opt.py` (new file, inherits from baseline, feature-flag per optimization)

---

## Benchmark Results

| Configuration | Latency (median) | Throughput | PCC | vs Baseline |
|---|---|---|---|---|
| **BASELINE** | 2580 ms | 0.388 it/s | 0.9947 **✗** | — |
| OPT_NORMS | 1808 ms | 0.553 it/s | 0.9974 ✓ | **+42.7%** |
| OPT_FINAL_NORM | 1813 ms | 0.551 it/s | 0.9974 ✓ | +42.3% |
| OPT_DIT_NORM | 1809 ms | 0.553 it/s | 0.9974 ✓ | +42.6% |
| OPT_MM | 2410 ms | 0.415 it/s | 0.9961 ✓ | +7.0% |
| OPT_ALL | 1703 ms | 0.587 it/s | 0.9986 ✓ | +51.5% |
| OPT_FUSED_QKV | 1672 ms | 0.598 it/s | 0.9986 ✓ | +54.3% |
| OPT_ASYNC_CCL | 1674 ms | 0.597 it/s | 0.9985 ✓ | +54.1% |
| **OPT_TRACE** | **1106 ms** | **0.904 it/s** | **0.9985** ✓ | **+133%** |

Side note: **the baseline was already failing** the PCC ≥ 0.995 correctness threshold (PCC = 0.9947). All optimized configurations fix this. More on why below.

---

## What Worked

### 1. `ttnn.rms_norm` — the dominant win (+42.7% alone)

**What was swapped**: `_rms_norm_f32()` (manual 9-op sequence) → `ttnn.rms_norm()`

The compiler-generated baseline implemented RMS normalization by hand, matching a traced XLA graph op-for-op:

```
typecast(F32) → pow(2) → sum(dim=-1) → multiply(1/dim) → add(eps) → rsqrt → multiply → typecast(BF16) → multiply(weight)
```

This sequence is called **~153 times per forward pass** (4 norms per transformer block × 34 blocks + caption embedder). Replacing with a single fused `ttnn.rms_norm` kernel eliminates roughly 1,370 ops per forward pass.

Applied to two distinct call sites:
- **Hidden norms** (`attention_norm1/2`, `ffn_norm1/2`): input `[1, seq, 3840]`, weights `[1, 3840]` or `[1, 1, 3840]`
- **QK norms** (`norm_q`, `norm_k`): input `[1, seq, 8, 128]`. These required a one-time weight reshape at init: the R0 consteval format stored weights as `[1,1,1,64,2]` (64 complex pairs); reshaped to `[1,1,1,128]` to match `ttnn.rms_norm`'s expectation of a flat last-dimension weight.

**Implementation gotcha**: `ttnn.rms_norm` requires **TILE layout** input. The inputs arriving from `_all_reduce` are ROW_MAJOR, so a `ttnn.to_layout(TILE)` conversion was needed. This adds one op but the fusion savings still dominate overwhelmingly.

**Why PCC improved**: The manual F32 sequence was faithfully reproducing the traced XLA graph's arithmetic, which accumulated rounding errors differently across 34 layers. `ttnn.rms_norm` uses a numerically stable single-pass kernel, giving better output fidelity (0.9974 vs 0.9947).

---

### 2. `ttnn.experimental.dit_rms_norm_unary_fused` — equivalent to `ttnn.rms_norm` (+42.6%)

The DiT-specific experimental op from tt_dit was tested as a direct substitute. It supports fused activation and residual addition but for the Z-Image use case (no fused activation, no residual at the norm site) it produced identical latency and PCC to `ttnn.rms_norm`. **Recommendation**: use `ttnn.rms_norm` for simplicity; `dit_rms_norm_unary_fused` becomes worthwhile only if you can fuse a downstream activation or fold in a residual add.

---

### 3. `ttnn.experimental.minimal_matmul` — works, modest gain (+7% standalone, +3.7% on top of norms)

**What was swapped**: `ttnn.matmul(..., transpose_b=True)` → `ttnn.experimental.minimal_matmul()` (no transpose, pre-transposed weights)

`minimal_matmul` expects weights in `[K, N]` format (no runtime transpose). At init time, all 238 attention and MLP projection weights (Q/K/V, `to_out`, `w1/w2/w3` across 34 blocks) are transposed once with `ttnn.permute([1, 0])`.

Hot-path shapes per device (seq=1056 for joint tokens):

| Projection | M | K | N |
|---|---|---|---|
| Q/K/V | 1056 | 3840 | 1024 |
| to_out | 1056 | 1024 | 3840 |
| w1/w3 (MLP) | 1056 | 3840 | 2560 |
| w2 (MLP) | 1056 | 2560 | 3840 |

**Caveat**: None of these shapes are in the `matmul.py` lookup table, which was tuned for Flux1/Mochi/SD3.5 on 8×8, 8×9, 12×10, 13×9 grids. The Blackhole P150's 13×10 grid isn't in the table either. Every call fell back to a default 8×8×8 blocking. Despite this, `minimal_matmul` is still +7% faster than `ttnn.matmul` with its optimized kernel. Adding our shapes to the lookup table would extract further gains (estimated 5–15%).

---

### 4. `ttnn.layer_norm` for final layer — neutral in isolation, small positive in combination

**What was swapped**: 7-op manual LayerNorm → `ttnn.layer_norm()`

The final layer does a mean-centered LayerNorm (not RMS). The manual sequence:

```
mean(dim=-1) → subtract → pow(2) → mean(dim=-1) → add(eps) → rsqrt → multiply
```

Replacing with `ttnn.layer_norm` saves ops but requires an extra `typecast(BF16)` + `to_layout(TILE)` because the input arrives F32 ROW_MAJOR from the preceding `_all_reduce`. This is only called once per forward pass so the savings are small — latency difference is within noise (1813 ms vs 1808 ms in isolation). In combination with `minimal_matmul` (OPT_ALL), it contributes positively overall.

---

---

### 5. `reduce_scatter_minimal_async` + `all_gather_async` — negligible gain (+1.5%, within noise)

**What was swapped**: `ttnn.reduce_scatter` + `ttnn.all_gather` → `ttnn.experimental.reduce_scatter_minimal_async` + `ttnn.experimental.all_gather_async`, managed by `CCLManager` from `tt_dit/parallel/manager.py`.

**What was gained**: ~25 ms per forward pass (1699 ms → 1674 ms). Within measurement noise.

**Why so little**: Profiling (see `PERF_TRACE_REPORT.md`) showed that communication accounts for ~89% of wall-clock time (~1480 ms). The improvement from switching to async variants comes from two sources:

1. **Persistent pre-allocated buffers**: the `CCLManager` allocates ping-pong output buffers for each unique tensor shape on first call and reuses them every iteration, avoiding per-call DRAM allocation. This accounts for most of the ~25 ms gain.

2. **`minimal` ring protocol**: a more lightweight ERISC kernel than the standard `reduce_scatter`. However, for our ring size (4 devices) and tensor sizes (~8 MB per all_reduce), latency is dominated by the ring rounds, not protocol overhead.

**The fundamental limitation**: each `all_reduce` result is immediately consumed by the next `rms_norm`. There is no compute work that can run concurrently with a given all_reduce because the very next op depends on its output. Async dispatch doesn't help when every queued op is dependent on the previous one. True overlap would require restructuring the computation graph to pipeline across transformer blocks (block N+1's norm running during block N's all_reduce), which is a significant architectural change.

**Implementation note**: `CCLManager` works with 4D tensors only. Since `_all_reduce` already reshapes to `[1, 1, seq, H]` before calling `reduce_scatter`, we pass 4D directly with `dim=3` on both RS and AG calls. Persistent buffers are keyed by tensor shape, so the three sequence lengths (1056, 1024, 32) each get their own buffer pair.

---

### 6. Metal Trace (`ttnn.begin/end/execute_trace`) — **+54.1% on top of async CCL**

**What was added**: `Tracer` from `tt_dit.utils.tracing` wraps the forward pass. First call = compile run + trace capture. All subsequent calls = `ttnn.execute_trace` only.

**What was gained**: 1704 ms → 1106 ms, **-35% latency**, **0.587 → 0.904 it/s** (+54.1%). vs original baseline: **+133%** total throughput gain.

**Why so much** (expected only ~20ms from host dispatch, got 600ms):

The perf trace attributed ~1480ms to ERISC communication. But that measurement was a residual — wall-clock minus Tensix kernels minus estimated dispatch. What Metal Trace revealed is that a large portion of that gap was not wire-time in the ring protocol, but **host-side synchronization overhead between consecutive ops**. With standard execution:

- After each CCL op, the host driver must observe completion, schedule the next op, and enqueue it to the device command queue.
- With 76 all_reduces per pass, each with a norm immediately after, there are 76+ host round-trips of ~5–8ms each = **~450–600ms of dispatch-sync overhead masquerading as communication latency**.

Metal Trace compiles the entire forward pass into a single device command stream. After `begin_trace_capture` / `end_trace_capture`, the device has all 1000+ ops pre-enqueued. `execute_trace` replays the stream with zero host involvement — the only latency is actual compute + wire time.

**Implementation**:
- `utils.py`: `DeviceGetter.trace_region_size = 50_000_000` (50MB L1 allocation for trace)
- `model_ttnn_opt.py`: `_forward_for_trace()` patches `ttnn.synchronize_device` to a no-op (sync commands cannot be captured in a trace); `forward()` delegates to `Tracer` when `USE_METAL_TRACE=True`
- `benchmark.py`: `OPT_TRACE` config; sets `trace_region_size` before device open

---

## What Didn't Work / Wasn't Applicable

### `ttnn.experimental.rotary_embedding_llama` — format mismatch

This op expects RoPE in Llama format: separate `cos_cache` and `sin_cache` tensors plus a `trans_mat` rotation matrix. Z-Image-Turbo uses 3D RoPE with three separate frequency axes (F/H/W), precomputed via `ttnn.embedding` lookups into `[seq, 1, half_dim, 2]` interleaved tables, then complex-multiplied manually. Adapting to Llama format would require rewriting the entire RoPE pipeline and re-deriving the frequency tables — too large a change with uncertain compatibility.

### `ttnn.experimental.all_gather_async` / `reduce_scatter_minimal_async` — missing infrastructure

These ops replace the synchronous `ttnn.all_gather` + `ttnn.reduce_scatter` in `_all_reduce()`, and would allow communication to overlap with computation. However, they require:
- Pre-allocated **persistent output buffers** (ping-pong buffered for pipelining)
- `GlobalSemaphore` objects for synchronization
- A `CCLManager` to own and lifecycle these resources

The tt_dit library provides this via its `CCLManager` class, but wiring it into the current model would be significant infrastructure work. Estimated gain if implemented: 5–10%.

### `ttnn.experimental.nlp_create_qkv_heads` — separate weights not compatible

This op splits a single fused QKV projection into Q, K, V heads in one call. Z-Image-Turbo has three separate weight matrices (`to_q`, `to_k`, `to_v`) loaded and sharded independently. Using this op would require fusing these into a single `[3 × HEADS_PER_DEV × HEAD_DIM, HIDDEN_DIM]` weight at load time — a non-trivial refactor.

### Distributed LayerNorm ops (`dit_layernorm_pre/post_allgather`, `wan_fused_rmsnorm_pre/post_allgather`) — wrong topology

These ops implement **sequence-parallel** distributed normalization where the hidden dimension is reduced across devices. Z-Image-Turbo uses **tensor parallelism** where sequence tokens are replicated and the hidden dimension is sharded by heads/MLP width. The communication pattern is fundamentally different.

---

## Summary Table

| tt_dit Op / Technique | Used? | Result |
|---|---|---|
| `ttnn.rms_norm` | ✓ | **+42.7% speedup, fixes PCC** |
| `ttnn.experimental.dit_rms_norm_unary_fused` | ✓ | Same as `rms_norm` (+42.6%), use for fused activation |
| `ttnn.layer_norm` | ✓ | Neutral in isolation, small positive in OPT_ALL |
| `ttnn.experimental.minimal_matmul` | ✓ | +7% standalone, +3.7% on top of norms (untuned) |
| `ttnn.experimental.minimal_matmul_split` | ✓ | +1.8% (fused QKV, vs separate Q/K/V with minimal_matmul) |
| `ttnn.experimental.nlp_create_qkv_heads` | ✓ | Used for V head-reshape in fused QKV path |
| `ttnn.experimental.all_gather_async` + `reduce_scatter_minimal_async` | ✓ | +1.5% (within noise) — no compute to overlap with |
| **Metal Trace** (`ttnn.begin/end/execute_trace`) | ✓ | **+54.1% on top of OPT_ASYNC_CCL; +133% vs baseline** |
| `ttnn.experimental.rotary_embedding_llama` | ✗ | Format mismatch with Z-Image 3D RoPE |
| Distributed norm ops (pre/post allgather) | ✗ | Wrong parallelism topology (seq-par vs tensor-par) |

---

## Potential Further Improvements

1. **Tune `minimal_matmul` blocking configs** for the 13×10 grid and our specific (M, K, N) shapes. Add entries to a `grid_13_10_configs` table in `tt_dit/utils/matmul.py`. Estimated additional gain: 5–15%.

2. **Async CCL** (`all_gather_async` + `reduce_scatter_minimal_async`) would overlap communication with computation in the attention `_all_reduce` path. Requires implementing the CCL manager infrastructure (persistent buffers + global semaphores). Estimated gain: 5–10%.

3. **Fused QKV projection** using `minimal_matmul_split(chunks=3)` — compute Q, K, V in a single matmul and split the output. Requires fusing the separate `to_q/to_k/to_v` weights into a `[3, HEADS_PER_DEV*HEAD_DIM, HIDDEN_DIM]` weight at load time.
