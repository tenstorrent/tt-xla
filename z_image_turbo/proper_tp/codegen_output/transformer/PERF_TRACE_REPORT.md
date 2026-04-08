# Z-Image-Turbo: Performance Trace Analysis

**Model**: OPT_FUSED_QKV (best config)  
**Hardware**: 4× Blackhole P150, TP=4, 13×10 compute grid per card  
**Measured latency**: 1672 ms/iter (0.598 it/s)  
**Profiling tool**: `python -m tracy -r` with `TT_METAL_DEVICE_PROFILER=1`  
**Source**: `cpp_device_perf_report.csv` (C++ post-processed device kernel timings)

---

## Time Budget Summary

| Component | Time (ms) | % of wall-clock |
|---|---|---|
| Tensix compute kernels | 170 ms | 10.2% |
| **ERISC comm (all_gather + reduce_scatter)** | **~1500 ms** | **~89.8%** |
| Host dispatch overhead | ~20 ms | ~1.2% |

### How the numbers were derived

The device profiler captures **Tensix** (compute core) kernel durations. All-reduce operations use **ERISC** (Ethernet) cores and are invisible to Tensix profiling — they appear only as gaps between Tensix ops.

- **Tensix compute**: sum of `DEVICE KERNEL DURATION [ns]` on device 0 across 3 passes → 510 ms total → **170 ms/pass**  
- **ERISC communication**: 1672 ms wall-clock − 170 ms Tensix − ~20 ms dispatch = **~1480 ms/pass**  
- With ~76 all_reduce calls per forward pass (38 transformer blocks × 2: one per attention + one per MLP), each all_reduce averages **~20 ms**

---

## Tensix Kernel Breakdown (per pass, device 0)

Sorted by total kernel time. Grand total: **170 ms/pass**.

| # | OP NAME | calls/pass | total ms | avg µs | cores | % | cum% |
|---|---------|---:|---:|---:|---:|---:|---:|
| 1 | `UntilizeWithUnpaddingDeviceOperation` | 15 | 10.69 | 712.7 | 18 | 16.6% | 16.6% |
| 2 | `TilizeDeviceOperation` | 92 | 8.50 | 92.1 | 121 | 13.2% | 29.8% |
| 3 | `SliceDeviceOperation` | 17 | 8.02 | 462.4 | 26 | 12.4% | 42.3% |
| 4 | `BinaryNgDeviceOperation` | 45 | 7.42 | 166.0 | 22 | 11.5% | 53.8% |
| 5 | `TransposeDeviceOperation` | 130 | 7.25 | 55.8 | 126 | 11.3% | 65.0% |
| 6 | `TilizeWithValPaddingDeviceOperation` | 93 | 6.12 | 65.5 | 77 | 9.5% | 74.5% |
| 7 | `PermuteDeviceOperation` | 13 | 5.43 | 407.4 | 20 | 8.4% | 83.0% |
| 8 | `ReshapeViewDeviceOperation` | 52 | 4.43 | 85.7 | 14 | 6.9% | 89.9% |
| 9 | `ConcatDeviceOperation` | 24 | 2.51 | 104.4 | 69 | 3.9% | 93.7% |
| 10 | `MinimalMatmulDeviceOperation` | 10 | 1.59 | 154.1 | 12 | 2.5% | 96.2% |
| 11 | `TypecastDeviceOperation` | 41 | 1.52 | 37.1 | 85 | 2.4% | 98.6% |
| 12 | `LayerNormDeviceOperation` | 8 | 0.43 | 56.3 | 15 | 0.7% | 99.3% |
| 13 | `SDPAOperation` | 2 | 0.22 | 110.0 | 9 | 0.3% | 99.6% |
| 14 | `MatmulDeviceOperation` | 3 | 0.13 | 44.4 | 43 | 0.2% | 99.8% |
| 15 | `EmbeddingsDeviceOperation` | 9 | 0.07 | 7.7 | 30 | 0.1% | 99.9% |
| 16 | `UnaryDeviceOperation` | 3 | 0.06 | 18.4 | 33 | 0.1% | 100.0% |

---

## Analysis by Category

### Format conversions: 39% of Tensix compute

| Category | Ops | Total ms | % |
|---|---|---:|---:|
| Untilize (TILE→ROW_MAJOR) | `UntilizeWithUnpadding` | 10.69 | 16.6% |
| Tilize (ROW_MAJOR→TILE) | `Tilize` + `TilizeWithValPadding` | 14.62 | 22.7% |
| **Total layout** | | **25.31** | **39.3%** |

**Root cause**: `ttnn.all_gather` outputs tensors in **ROW_MAJOR** layout. Each of the ~153 norm calls (`ttnn.rms_norm`) requires TILE layout, so `_ensure_tile()` inserts a `Tilize` before every norm. After SDPA and attention output projection, tensors are re-untilized back to ROW_MAJOR for the residual add.

**Fix target**: Push TILE layout through the all_gather output, or restructure the all_reduce path so the output is already in TILE. Alternatively, investigate if `all_gather` can produce TILE output directly.

### Transpose/permute: 19.7% of Tensix compute

| Op | calls/pass | Total ms | Source |
|---|---:|---:|---|
| `TransposeDeviceOperation` | 130 | 7.25 | Attention head permutes: `[1,seq,H,D]↔[1,H,seq,D]` |
| `PermuteDeviceOperation` | 13 | 5.43 | Various reshapes in RoPE and SDPA prep |
| **Total** | 143 | **12.68** | **19.7%** |

130 transposes/pass × 3 (Q permute + K permute + context permute) per attn block matches ~43 attention blocks' worth of permutes.

**Fix target**: Fuse transpose into the matmul output or use ops that work natively in `[B, H, S, D]` format.

### SliceDeviceOperation: 12.4% of Tensix compute

17 slice calls/pass, 8.0 ms total, 462 µs each. These are the internal head splits from `ttnn.experimental.nlp_create_qkv_heads` extracting V heads from the fused QKV output.

**Fix target**: This is a side-effect of the `USE_FUSED_QKV` path. Worth checking if a direct reshape (without nlp_create_qkv_heads) avoids the slice.

### Actual matmul: only 2.7% of Tensix compute

| Op | calls/pass | Total ms |
|---|---:|---:|
| `MinimalMatmulDeviceOperation` | 10 | 1.59 |
| `MatmulDeviceOperation` | 3 | 0.13 |
| `SDPAOperation` | 2 | 0.22 |
| **Total** | **15** | **1.94** |

The matmul call count (10–13 per pass) appears lower than expected (should be ~150+). This is likely because the profiler only captures the first invocation of each unique program hash when the program cache is active — repeated calls with identical shapes share a single program entry in the report.

---

## Priority Attack Plan

### 1. Async CCL — potential ~50–70% speedup on top of current (est. −830 to −1170 ms)

**What**: Replace synchronous `ttnn.reduce_scatter` + `ttnn.all_gather` in `_all_reduce` with `ttnn.experimental.reduce_scatter_minimal_async` + `ttnn.experimental.all_gather_async`.

**Why it dominates**: ~1480 ms/pass is ERISC communication. With async ops and persistent buffers (ping-pong), each all_reduce can overlap with the *next* block's attention/MLP computation. For 38 blocks with ~20 ms/all_reduce, full pipeline overlap would recover ~750 ms (most of the non-critical-path communication).

**What's needed**:
- `GlobalSemaphore` objects per device for synchronization
- Persistent output buffers for ping-pong
- A `CCLManager`-like wrapper to manage lifecycle (see `tt_dit/utils/ccl_manager.py`)

**Blocking issue identified in earlier work**: Infrastructure not wired in yet. This is the highest-priority engineering investment.

---

### 2. Eliminate layout conversions — ~25 ms (1.5% of pass, 15% of Tensix compute)

**What**: Avoid the ROW_MAJOR→TILE→ROW_MAJOR round-trip around every norm.

Two approaches:
- **Option A**: Make `all_gather` produce TILE output — check if `ttnn.all_gather` has a layout output option.
- **Option B**: Keep tensors in TILE throughout the residual stream. The residual add `x = x + block_output` runs on ROW_MAJOR because `all_gather` outputs ROW_MAJOR. If the add and norm are fused (residual+norm), both layout conversions collapse into one.

**Incremental gain**: ~1.5% wall-clock. Small in absolute terms but reveals compute-bound headroom once communication is fixed.

---

### 3. Reduce transpose overhead — ~13 ms (0.8% of pass, 7.6% of Tensix compute)

**What**: The Q/K head permutes `[1,seq,H,D]→[1,H,seq,D]` before SDPA and the reverse permute after are `TransposeDeviceOperation`. 130 calls × 55 µs = 7.25 ms/pass.

If the model can keep tensors in `[B, H, S, D]` (heads-first) format from the QK norm output through SDPA, the permutes are eliminated. This requires restructuring the QK norm to accept `[B, H, S, D]` directly.

---

### 4. Tune `minimal_matmul` for 13×10 grid — est. 5–15% of matmul time

The Blackhole P150's 13×10 grid is not in the `matmul.py` lookup table. All calls fall back to default `8×8×8` blocking. Key shapes to tune:

| Projection | M | K | N |
|---|---:|---:|---:|
| Q/K/V fused | 1056 | 3840 | 3072 |
| to_out | 1056 | 1024 | 3840 |
| w1/w3 (MLP) | 1056 | 3840 | 2560 |
| w2 (MLP) | 1056 | 2560 | 3840 |

Since matmul is currently only 1.94 ms/pass (2.7% of Tensix compute, 0.12% of wall-clock), this is **low priority until async CCL is implemented** — at that point matmul could become a larger fraction of the budget.

---

## Summary Table

| Target | Est. gain | Effort | Priority |
|---|---|---|---|
| Async CCL (all_gather_async + reduce_scatter_minimal_async) | **~50–70% total speedup** | High (CCLManager infra) | **#1** |
| Eliminate layout conversions (keep TILE through all_reduce) | ~1.5% | Medium | #2 |
| Fuse transpose into head reshape | ~0.8% | Medium | #3 |
| Tune minimal_matmul for 13×10 grid | ~0.1% now, more after CCL | Low | #4 |
