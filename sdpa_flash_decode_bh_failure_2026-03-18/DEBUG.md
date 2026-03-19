# sdpa / sdpa_flash_decode Kernel JIT Build Failure on Blackhole

**Date**: 2026-03-18
**Priority**: High
**Status**: Root-caused, waiting on tt-metal/SFPI fix
**Affected hardware**: Blackhole (p100, p150a)
**GitHub issue**: https://github.com/tenstorrent/tt-xla/issues/3803
**CI failure**: https://github.com/tenstorrent/tt-xla/actions/runs/23262443567/job/67636614882

## Summary

All LlamaForCausalLM models fail during vLLM engine initialization on Blackhole. Both `sdpa` and `sdpa_flash_decode` tt-metal kernels fail to JIT compile. The root cause is `_calculate_exponential_approx_()` (introduced in tt-llk PR #1190, commit `7e7cf4fc`, Feb 8 by Stephen Osborne) triggering a GCC 15.1.0 codegen bug when compiled with `-O3 -flto=auto`.

## Error

```
sfpi.h:766:7: error: cannot write sfpu vector to memory
  766 |     v = (initialized) ? __builtin_rvtt_sfpassign_lv(v, in) : in;
```

The `sfpi::reinterpret` calls at lines 97 and 101 of `_calculate_exponential_approx_` go through `__vBase::assign()` in sfpi.h. LTO's aggressive optimization causes the compiler to attempt spilling an SFPU register to memory, which SFPU hardware cannot do.

Full failure is in the trisc1 link step with `-O3 -flto=auto -ffast-math -mcpu=tt-bh-tensix`.

## Root Cause Analysis

The bug is in the interaction between:
1. **`_calculate_exponential_approx_()`** in `ckernel_sfpu_exp.h` — uses `sfpi::reinterpret<vInt>` and `sfpi::reinterpret<vFloat>` which go through `__vBase::assign()`
2. **`__vBase::assign()`** in `sfpi.h:766` — the ternary `(initialized) ? __builtin_rvtt_sfpassign_lv(v, in) : in` generates code for both branches
3. **LTO (`-flto=auto`)** at link time — aggressively optimizes/clones the function, creating code patterns that require more SFPU registers than available, forcing a spill to memory

The bug does NOT exist without LTO. LTO was added to tt-metal JIT in June 2025 (PR #21878, `build.cpp:147`). The failing function was added in Feb 2026 (PR #1190). The combination was never tested on Blackhole.

## Workarounds Attempted (all failed)

| Approach | Result |
|----------|--------|
| `__attribute__((noinline))` | LTO still optimizes within function body. Got past `sdpa_flash_decode` but same bug in `sdpa`. |
| `__attribute__((noipa))` | Prevents LTO cloning (`constprop.isra` gone), but function itself still fails at `-O3` |
| `__attribute__((optimize("no-lto")))` | SFPU compiler doesn't support this attribute — build error |
| `__attribute__((optimize("O2")))` | Ignored during LTO link phase (global `-O3` overrides) |
| `asm volatile("" ::: "memory")` barriers | Ignored by LTO |

None of these work because the bug is in `sfpi.h`'s `assign()` codegen, not in `ckernel_sfpu_exp.h`. No function-level attribute can prevent LTO from optimizing the sfpi.h template code.

## Possible Fixes (for tt-metal/SFPI team)

1. **Restructure `_calculate_exponential_approx_`** to avoid the `sfpi::reinterpret` pattern that triggers the bug
2. **Remove `-flto=auto`** from JIT build flags (`build.cpp:147`) — simplest but affects all kernel performance
3. **Fix the SFPI compiler** codegen for `__vBase::assign()` under LTO

## Reproduction

Reproduced with Llama-3.2-3B, Llama-3.2-1B, and TinyLlama-1.1B — all hit the same failure. OPT-125M is unaffected (different attention path).

```bash
pytest -svv tests/integrations/vllm_plugin/generative/test_llama3_3b_generation.py
```

## Version Info

| Component | Version | Notes |
|-----------|---------|-------|
| tt_llk | `3790a3ed` (Feb 25) | Last change to exp.h: `7e7cf4fc` (Feb 8, PR #1190) |
| tt-metal | Mar 1 commit | LTO enabled since Jun 2025 |
| SFPI | 7.27.0 (Feb 27) | GCC 15.1.0, matches `sfpi-version` |
| Hardware | Blackhole | p100, p150a |

## Key Files

- **Failing function**: `tt_metal/third_party/tt_llk/tt_llk_blackhole/common/inc/sfpu/ckernel_sfpu_exp.h:88` (`_calculate_exponential_approx_`)
- **Root cause**: `/opt/tenstorrent/sfpi/include/sfpi.h:766` (`__vBase::assign`)
- **LTO flag**: `tt_metal/jit_build/build.cpp:147` (`-flto=auto`)
- **SFPI version**: `tt_metal/sfpi-version` (7.27.0)

## Call Chain

1. vLLM `llm = vllm.LLM(...)` → engine initialization
2. `compile_or_warm_up_model` → `capture_model` → `_precompile_backbone` → `_dummy_run`
3. torch dynamo → XLA → tt-mlir → tt-metal
4. tt-metal JIT compiles `sdpa` / `sdpa_flash_decode` kernels
5. trisc1 link fails on `_calculate_exponential_approx_` in BH exponential LLK

## Parameter Sweep Findings

The bug is **determined entirely by `num_query_heads`**. Sequence length, batch size, weight dtype (bfp8), kv_heads, and head_dim are all irrelevant.

### SDPA Op-Level Sweep (`test_sdpa_decode_sweep.py`)

| query_heads | kv_heads | head_dim | Result |
|---|---|---|---|
| 4 | 4 | 64 | PASS |
| 4 | 4 | 128 | PASS |
| 8 | 4 | 64 | PASS |
| 8 | 8 | 64 | PASS |
| 8 | 8 | 128 | PASS |
| 16 | 4 | 64 | PASS |
| 16 | 8 | 64 | PASS |
| 16 | 16 | 64 | PASS |
| 16 | 16 | 128 | PASS |
| 20 | 4 | 64 | **FAIL** |
| 24 | 8 | 128 | **FAIL** |
| 32 | 4 | 64 | **FAIL** |
| 32 | 8 | 64 | **FAIL** |
| 32 | 8 | 128 | **FAIL** |
| 32 | 16 | 64 | **FAIL** |
| 32 | 32 | 64 | **FAIL** |

**Threshold: `num_query_heads > 16` triggers the bug for all valid head combos.**

### vLLM Model Tests

| Model | Params | Query heads | Result | tok/s |
|---|---|---|---|---|
| **Qwen2.5-0.5B-Instruct** | 0.5B | 16 | **PASS** | 3.51 |
| **Qwen2.5-3B-Instruct** | 3B | 16 | **PASS** | 3.23 |
| Qwen2.5-7B-Instruct | 7B | 28 | FAIL (sdpa_flash_decode) | — |
| Llama-3.2-3B-Instruct | 3B | 24 | FAIL (sdpa) | — |
| Llama-3.1-8B-Instruct | 8B | 32 | FAIL (sdpa) | — |
| TinyLlama-1.1B | 1.1B | 32 | FAIL (sdpa) | — |
| Gemma-2-9B-it | 9B | 8 | FAIL (different error, not SDPA) | — |
| Qwen3-0.6B | 0.6B | 16 | PASS (known from other tests) | — |

### Seq Len / Weight Dtype Do Not Help (Llama-3.2-3B)

Tested with `max_model_len=32` (smallest possible) — still fails. The SDPA kernel template is parameterized by head count, not sequence length. bfp8 weight dtype also does not affect the SDPA kernel compilation.

Logs: `test_llama_sweep_seq32.log`, `test_llama_sweep_bfp8.log`

### Why Specific Head Counts Trigger the Bug

The SDPA kernel is templated on `num_query_heads`. Higher head counts create more complex loop unrolling in the exponential computation, increasing register pressure. At >16 heads, LTO's optimization of the `sfpi::reinterpret` → `__vBase::assign()` chain exceeds the available SFPU registers, forcing a spill.

### Full Baseline Run (no fix applied)

Log: `bh_single_device_vllm_no_fix.log`

| Test | Query heads | Result |
|---|---|---|
| `test_opt_generation` (OPT-125M) | 12 | PASS |
| `test_opt_generation_multibatch` | 12 | PASS |
| `test_responses_api_basic[string_input]` | 12 | PASS |
| `test_responses_api_basic[message_input]` | 12 | PASS |
| `test_responses_api_streaming` | 12 | PASS |
| `test_responses_api_deterministic` | 12 | PASS |
| `test_responses_api_instructions` | 12 | PASS |
| `test_responses_api_top_logprobs` | 12 | PASS |
| `test_vllm_generation[opt-125m]` | 12 | PASS |
| `test_vllm_generation[qwen2.5-0.5b]` | 16 | PASS |
| `test_vllm_generation[qwen2.5-1.5b]` | 12 | PASS |
| `test_vllm_generation[qwen2.5-3b]` | 16 | PASS |
| `test_llama3_3b_generation` (Llama-3.2-3B) | 24 | **FAIL** (sdpa) |
| `test_tinyllama_generation` | 32 | **FAIL** (sdpa) |
| `test_vllm_generation[tinyllama-1.1b]` | 32 | **FAIL** (sdpa) |
| `test_vllm_generation[llama3.2-1b]` | 32 | **FAIL** (sdpa) |
| `test_vllm_generation[llama3.2-3b]` | 24 | **FAIL** (sdpa) |
| `test_vllm_generation[qwen2.5-7b]` | 28 | **FAIL** (sdpa) |
| `test_vllm_generation[llama3.1-8b]` | 32 | **FAIL** (sdpa) |

All 7 failures are `cannot write sfpu vector to memory` in `sdpa_flash_decode`, all >16 query heads.

### Compiler Wrapper Workaround Attempted

Replaced `/opt/tenstorrent/sfpi/compiler/bin/riscv-tt-elf-g++` with a wrapper script that strips `-flto=auto` for sdpa kernels only.

- **`-O3 -fno-lto`**: Kernel compiles but binary too large (71296 > 70688 byte Tensix limit)
- **`-O3 -fno-lto -ffunction-sections -fdata-sections -Wl,--gc-sections`**: Kernel compiles and fits, but produces **garbage output** (attention computation incorrect without LTO)
- **`-O3 -fno-lto + sfpi.h v=in hack`**: Same garbage output — the sfpi.h change is not the cause

Conclusion: SDPA kernel on BH requires LTO for both fitting in the Tensix buffer and correct codegen. Compiler-level workarounds are not viable.

### Fixes

Two fix paths identified:

1. **tt-mlir PR [#7561](https://github.com/tenstorrent/tt-mlir/pull/7561)**: Disables `exp_approx_mode` for BH paged SDPA decode in tt-mlir runtime. Falls back to precise exp which doesn't trigger the bug. In our control.

2. **tt-metal PR [#39474](https://github.com/tenstorrent/tt-metal/pull/39474)**: Changes SDPA to no longer call `_calculate_exponential_piecewise_`. Cherry-picked to tt-metal branch `kmabee/metal_march18_exp_approx_sdpa_cherry_pick` and confirmed working.

### Demo Workaround (without fix)

**Qwen2.5-3B-Instruct** (16 query heads, 3B params) is the largest model confirmed working via vLLM on single-chip Blackhole without any fix. Serving example at `examples/vllm/Qwen2.5-3B-Instruct/`.
