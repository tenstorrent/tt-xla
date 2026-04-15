# Context: Extending ttnn.topk for large vocab sizes

## Problem

`ttnn.topk` multi-core mode requires input dimension to be power-of-2 AND < 65536 (uint16 index constraint). For LLM vocab sizes like 128256 (Llama-3.1), this forces a 4x chunking strategy in the sampler that generates ~30 compiled TTNN ops. Each op has ~0.5ms host-side dispatch overhead in the tt-xla program executor, adding ~15ms/token and limiting non-greedy sampling to 13.4 tok/s vs 19.0 greedy.

If `ttnn.topk` supported input dims ≥65536 with multi-core, a single `torch.topk(logits, k=32)` on the full 128K vocab would compile to 1 TTNN op instead of ~30, closing most of the performance gap.

## Current constraints (from topk_device_operation.cpp)

Multi-core execution requires ALL of:
1. Input dim ≥ `multi_core_min_width` (8192)
2. Input dim < 65536 (uint16 index limit) AND power-of-2
3. K ≤ 64
4. Memory/core constraints pass (L1 fits, cores divisible)

If any fails → falls back to single-core `ttnn.sort` (30ms on 128K vocab).

The uint16 constraint (requirement #2) is the hard blocker. It's baked into:
- `topk_device_operation.cpp:69`: `input_shape[args.dim] < std::numeric_limits<uint16_t>::max()`
- The multi-core program factory uses `UInt16` data format for index circular buffers
- The compute kernels likely use 16-bit index storage

## Key files in tt-metal

All under `ttnn/cpp/ttnn/operations/reduction/topk/`:

| File | Lines | Purpose |
|---|---|---|
| `device/topk_device_operation.cpp` | ~100 | Program factory selection (multi-core vs single-core) |
| `device/topk_multi_core_program_factory.cpp` | 485 | Multi-core program setup, CB allocation, kernel dispatch |
| `device/topk_single_core_program_factory.cpp` | 276 | Single-core fallback |
| `device/topk_constants.hpp` | 15 | `multi_core_min_width=8192`, `min_dim_per_core=64` |
| `device/topk_utils.cpp` | | Utility functions |
| `device/kernels/compute/` | | Compute kernels (bitonic sort, merge) |
| `topk.hpp` / `topk.cpp` | | Public API |

## What needs to change

The goal is to support input dims up to ~256K (covering all LLM vocabs) in multi-core mode. Two approaches:

### Approach A: Internal chunking in the device operation

Keep the existing multi-core kernel (which works great for ≤32768) but add chunking logic in `topk_device_operation.cpp` or a new program factory:

1. When dim > 32768: split into power-of-2 chunks ≤ 32768
2. Run existing multi-core topk on each chunk
3. Merge chunk results with a final topk on the merged candidates
4. Map local indices back to global positions

This is less invasive — the existing kernels stay the same. The chunking is orchestrated at the program factory level.

### Approach B: Extend the kernel to support uint32 indices

Modify the multi-core compute kernels and program factory to use uint32 indices when dim ≥ 65536. This is more work (kernel changes, double the index memory) but removes the fundamental constraint.

Key changes needed:
- `topk_device_operation.cpp`: remove/raise the uint16 limit
- `topk_multi_core_program_factory.cpp`: use `UInt32` index data format when needed, double index CB sizes
- Compute kernels: support 32-bit index operations in bitonic sort/merge
- Power-of-2 requirement might also need relaxing (128256 isn't power-of-2)

### Approach C: Hybrid — chunk to power-of-2, merge with uint16

Split 128256 → 4×32768 (power-of-2, fits uint16), run multi-core on each, merge with a final multi-core topk on the small merged set. All within the device operation.

This is approach A but explicitly keeping uint16 throughout. No kernel changes needed.

## Code owners

Most active contributors to topk (by substantive commits):
- **Miłosz Gajewski** (mgajewski@tenstorrent.com) — recent bug fixes, core range check, general updates
- **Djordje Ivanovic** (djordje-tt) — L1 space reduction in multi-core
- **Nathan Maurice** (nmaurice@tenstorrent.com) — cleanup and sweep tests
- **Artem Yerofieiev** (ayerofieiev-tt) — reduction ops migration, most commits overall

## Performance data

| Sampler config | tok/s | TTNN ops | Why |
|---|---|---|---|
| Greedy (argmax) | 19.0 | 3 | Target |
| 1x topk on 32K chunk | 18.5 | ~8 | 1 multi-core topk, fast |
| 4x topk on 4×32K chunks | 13.2 | ~30 | 4 multi-core topk + slices/pads/cats |
| Single topk on 128K | 12.3 | ~5 | Falls back to single-core sort (30ms) |

If a single multi-core topk on 128K took the same ~0.18ms as 32K (with uint32 indices), we'd get ~18.5 tok/s — matching greedy.

## Existing tests

- `tests/ttnn/unit_tests/operations/eltwise/test_sampling.py`
- `models/common/tests/test_sampling.py`
- Topk sweep tests (referenced in cleanup PR #37951)

## Branch / context

- tt-xla branch: `kmabee/vllm_perf_apr12`
- tt-mlir branch: `kmabee/apr12_vllm_demo_sampling_op_integration`
- Tracking issue: https://github.com/tenstorrent/tt-xla/issues/3940
- Hardware: P150 Blackhole, Llama-3.1-8B-Instruct
