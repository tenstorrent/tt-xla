## Update — April 15, 2026

### ttnn.sampling integration complete

Integrated `ttnn.sampling` as a fused custom op through the full tt-mlir pipeline (TTIR → TTNN → flatbuffer → runtime, 21 files on branch `kmabee/apr12_vllm_demo_sampling_op_integration`). This replaces the broken compiled gather/softmax/Gumbel-max chain with a single kernel that does softmax + top-k + top-p + multinomial on the reduced candidate set.

The v1 correctness bug from the last update is resolved — the root cause was double temperature application. The sampler was dividing logits by temperature before topk, then passing raw temperature to `ttnn.sampling` which multiplies by it internally (kernel expects `1/temperature`). Fixed by skipping the sampler's temperature step and passing `1/temperature` directly.

Output verified coherent on both OPT-125M and Llama-3.1-8B.

### Current performance (Llama-3.1-8B, batch=1)

| Config | tok/s | Notes |
|---|---|---|
| Greedy device | 19.0 | Baseline target |
| Non-greedy + ttnn.sampling | **13.4** | temp=0.6, no penalties |
| Non-greedy + penalties | 12.7 | temp=0.6, rep_penalty=1.1 |
| Non-greedy CPU sampling | 14.3 | |

Up from 5.1 tok/s baseline and 12.3 from the previous update. The v1 correctness bug is fixed, ttnn.sampling adds essentially zero overhead (0.27ms steady-state per tracy), and penalties add only 0.7 tok/s.

### Root cause of the remaining gap (19.0 → 13.4)

Confirmed via systematic experiments + IR analysis: the compiled non-greedy sampler graph has **50 dispatched TTNN ops**, each incurring ~0.5ms host-side dispatch overhead. Total: ~25ms/token.

The 50 ops break down as: 17 typecasts (bf16↔f32↔uint16 conversions for topk I/O), 6 pads (vocab alignment + batch-1→32 for sampling kernel), 5 slices (chunk extraction), 5 constants, 4 topk, 3 layout conversions, 3 adds (index offsets), 2 concats, 1 sampling, plus control flow ops. The greedy sampler has only 3 ops (to_layout + argmax + typecast).

Individual ops are fast — steady-state tracy shows topk at 0.077ms/call and sampling at 0.274ms/call. The bottleneck is the cumulative host→device dispatch latency, not device-side execution.

Key experiments confirming this:

| Sampler variant | tok/s | Compiled ops |
|---|---|---|
| Greedy (argmax) | 19.0 | 3 |
| Non-greedy argmax only | 18.6 | 3 |
| 1x topk (32K chunk) | 18.5 | ~8 |
| **4x topk (4×32K chunks)** | **13.2** | **~30** |
| Full ttnn.sampling | 13.4 | ~50 |

The jump from 1x topk to 4x topk accounts for almost the entire gap. The padding + ttnn.sampling ops on top of topk add negligible overhead.

### What's being explored next

Python-level restructuring can't reduce the op count — the compiler generates ~50 TTNN ops regardless of how the sampler Python code is organized (tested: separate compiled programs, batched reshape, bf16 fast path — none reduced effective dispatches).

**1. Reduce per-op dispatch overhead in the program executor** (root cause fix). Each of the 50 ops costs ~0.5ms in host-side dispatch overhead — far more than GPU kernel launch latency (~10μs). If this were reduced to ~50μs, the current 50-op sampler would add ~2.5ms instead of ~25ms, achieving ~18 tok/s with no changes to the sampler. This would benefit all compiled programs, not just sampling. Next step is profiling the program executor dispatch path to identify what's in that 0.5ms (flatbuffer deserialization? tensor pool lookups? synchronization?).

**2. Runtime-level chunked topk** (targeted fix). Modify the `ttnn.topk` runtime in tt-mlir to internally split large vocabs (>32K) into multi-core-friendly chunks, run per-chunk topk, and merge — all within a single dispatch. This reduces the sampler from ~50 compiled ops to ~10, targeting ~17-18 tok/s. The device-side work is identical, but direct C++ calls inside the runtime skip the per-op program executor overhead. A draft implementation exists. Downside: moves topk chunking logic into C++ runtime, reducing flexibility.

**3. Fused `tt::topk_sample` custom op** (most aggressive targeted fix). A new custom op that takes full-vocab logits + temperature and does the entire topk + sampling pipeline in one dispatch — same pattern as the existing `tt::sampling` integration. Reduces the sampler to ~5 compiled ops. Follows the established custom op integration path (21 files for `tt::sampling`). Downside: bakes the full sampling strategy into C++.

**4. Extend `ttnn.topk` in tt-metal to support input dims ≥65536** (cleanest fix). Currently multi-core topk requires power-of-2 input dim < 65536, forcing the 4x chunking. Native support for larger dims would let the sampler use a single `torch.topk` call on the full 128K vocab, compiling to 1 TTNN op. This eliminates chunking entirely and keeps the sampler as pure Python torch ops.
