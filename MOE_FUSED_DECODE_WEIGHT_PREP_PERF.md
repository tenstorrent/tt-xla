# Fused MoE decode: host-side const-eval weight-prep is the runtime bottleneck

**Context:** GPT-OSS fused MoE decode (`tt.moe_decode` → a2a + `moe_compute`) on the Wormhole
32-chip galaxy (tt-xla, branch `hshah/moe-compute-path`). Correctness is proven (20B 1-layer PCC
0.9999; native op tests pass). The remaining problem is purely **performance**: full-model / 120B
runs take 1–2+ hours, and this doc pins down why and how to fix it.

## TL;DR

The slow part is **NOT the fabric and NOT the tt-mlir compile.** It is **host-side weight
quantization executed at const-eval time** — `quantize_weights_via_host` (bf16→bfp4 on the host CPU,
single-threaded), run **once per layer, uncached, every process**.

- tt-mlir compile of the whole 24-layer 20B decode graph: **~45 seconds** (shlo→ttir→ttnn→flatbuffer,
  measured from IR dump mtimes). Compile is fine.
- MoE weight-prep quantize: **~34 min** for 24-layer 20B (measured, see below).
- Embed/lm_head weight-prep quantize (uninstrumented, inferred): **~50 min** more.
- These re-run on **every** invocation (no compile cache, no packed-weight cache).

## Where it happens (code paths)

The `moe_decode` composite lowers (in `TTNNResolveComposites.cpp`) to 4 TTNN ops **per layer**:
`all_to_all_dispatch_metadata`, `prepare_moe_compute_w0_w1_weights`, `prepare_moe_compute_w2_weights`,
`moe_compute`. The two `prepare_moe_compute_*` ops are **const-eval-hoisted**, so they execute during
the runtime const-eval phase (not compile).

Runtime wrappers (tt-mlir):
- `runtime/lib/ttnn/operations/ccl/prepare_moe_compute_w0_w1_weights.cpp`
- `runtime/lib/ttnn/operations/ccl/prepare_moe_compute_w2_weights.cpp`

Each does two tt-metal calls (`ttnn/cpp/ttnn/operations/experimental/ccl/moe_compute/moe_compute_utils.cpp`):
1. `prepare_w0_w1_tensor_for_moe_compute` / `prepare_w2_tensor_for_moe_compute` (+ `_with_bias`) — a
   pure layout/packing transform (pad K, tile, interleave W0/W1, permute, core-shard). **Fast.**
2. **`quantize_weights_via_host(packed, BFLOAT4_B, memcfg)`** — the hotspot. Its body:
   ```cpp
   auto host_tensor = ttnn::from_device(device_tensor);   // device -> host
   auto cast_tensor  = ttnn::to_dtype(host_tensor, dtype); // HOST bf16 -> bfloat4_b (single-threaded)
   auto result       = ttnn::to_device(cast_tensor, ...);  // host -> device (sharded)
   ```
   The `moe_compute` nanobind doc calls this the *"slower but higher quality"* path and names
   `ttnn.typecast` as the *"faster"* on-device alternative.

## Measurements (instrumentation)

Added `[MOE_PREP_TIMING]` fprintf(stderr) logging to both wrappers (prepare vs quantize seconds; also
reports the `fast` flag). From the 24-layer 20B run (EP-4 → E=8 experts/device, `fast=0`):

```
[MOE_PREP_TIMING] w0w1 L=1 E=8 bias=1 fast=0: prepare=~0.0s quantize=~55.4s
[MOE_PREP_TIMING] w2   L=1 E=8 bias=1 fast=0: prepare=~0.0s quantize=~28.5s
```
- **24 w0w1 + 24 w2 calls** (one per layer; NOT batched L=24), total quantize ≈ **2048s = 34.1 min**.
- **Quantize is ~96% of MoE weight-prep**; `prepare` (packing) is negligible.
- 1-layer 20B: prep ≈ 86s. Scales ~linearly with layers AND experts (120B has 4× the experts/device).

Stage timeline for the 24-layer 20B decode graph (from `modules/irs/` and `modules/*.ttnn` mtimes):
`vhlo/shlo/ttir 04:22:33-35 → ttnn + flatbuffer 04:23:18` (≈45s compile), then everything after
04:23:18 is **flatbuffer execution / const-eval** (MoE quantize 04:23–04:57, then a further ~50 min
of uninstrumented host const-eval — almost certainly the embed/lm_head bf16→bfp8 host quantize of the
201088×2880 tensors).

Not cached: `XLA_PERSISTENT_CACHE_PATH` unset, no persistent-cache code in tt-xla, ttnn
`enable_model_cache=false` → the whole prep re-runs every process.

## Concrete solutions (future work), roughly by leverage

1. **On-device quantize instead of host round-trip (already prototyped).** Replace
   `quantize_weights_via_host` with `ttnn::typecast(packed, BFLOAT4_B, memcfg)` (on-device). Prototyped
   behind env `TTXLA_MOE_FAST_QUANTIZE=1` in both `prepare_moe_compute_*.cpp` wrappers (built, **not yet
   A/B-tested**). Eliminates the from_device/host-to_dtype/to_device path. Trade-off: potential PCC/quality
   drop from on-device bf4 rounding — validate with the PCC test (compare `fast=1` vs `fast=0`). Apply the
   same swap to the embed/lm_head prep path.
2. **Persistent packed-weight cache.** Cache the bfp4-packed prepared weights to disk keyed by
   (weight tensor hash + shape + dtype + shard config); skip re-prep on subsequent runs. Biggest win for
   iterative runs (prep becomes one-time). tt-metal has an `enable_model_cache` knob to investigate; or add
   a tt-xla-level cache around the `prepare_moe_compute_*` runtime ops.
3. **Persistent PJRT/XLA compile cache.** Cache the compiled flatbuffer *including* the const-eval'd
   constants, so re-runs skip compile + all const-eval. Removes prep entirely on cache hit.
4. **Parallelize the host `to_dtype` (BFLOAT4_B).** The single-threaded host quantization is the raw
   hotspot; a multi-threaded block-float-4 packer would cut it ~Ncores×. Upstream tt-metal fix.
5. **Batch per-layer prep into one call (L=all_layers).** Currently 24 separate L=1 calls; batching to
   one L=24 call amortizes per-call from_device/to_device overhead (the quantize itself is O(elements) so
   the compute won't shrink, but the round-trip count drops 24×→1×).
6. **Instrument the embed/lm_head const-eval** to confirm it is the ~50-min post-MoE phase, then apply
   (1)/(4) there too (same `to_dtype`-on-host pattern on the 201088×2880 tensors).

## Files touched for the instrumentation + fast path (working-tree, tt-mlir submodule)
- `runtime/lib/ttnn/operations/ccl/prepare_moe_compute_w0_w1_weights.cpp` — `[MOE_PREP_TIMING]` + `TTXLA_MOE_FAST_QUANTIZE` gate.
- `runtime/lib/ttnn/operations/ccl/prepare_moe_compute_w2_weights.cpp` — same.
(These are NOT in `/home/hshah/moe-fused-decode-tt-mlir.patch`, which predates them — regenerate the patch to capture them.)
