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
