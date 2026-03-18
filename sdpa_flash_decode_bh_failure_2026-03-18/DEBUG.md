# sdpa_flash_decode Kernel JIT Build Failure on Blackhole

**Date**: 2026-03-18
**Priority**: High
**Status**: Investigating
**Affected hardware**: Blackhole only (passes on Wormhole)

## Summary

Llama-3.2-3B inference via vLLM fails during model precompilation. The tt-metal JIT compiler cannot build the `sdpa_flash_decode` kernel for Blackhole Tensix cores. The SFPI compiler hits a register spill that the SFPU hardware doesn't support.

## Error

```
sfpi.h:766:7: error: cannot write sfpu vector to memory
  766 |     v = (initialized) ? __builtin_rvtt_sfpassign_lv(v, in) : in;
```

Inlined from:
- `operator=` at `sfpi.h:342`
- `_calculate_exponential_piecewise_` at `ckernel_sfpu_exp.h:143`

Full failure is in the trisc1 link step of the `sdpa_flash_decode` kernel with `-O3 -flto=auto -ffast-math -mcpu=tt-bh-tensix`.

## Why Blackhole Only

- Uses `tt_llk_blackhole/common/inc/sfpu/ckernel_sfpu_exp.h` (BH-specific LLK)
- Wormhole uses `tt_llk_wormhole_b0/` which has a different exponential implementation
- The BH variant likely has higher register pressure, causing the spill

## Reproduction

```
pytest -svv tests/integrations/vllm_plugin/generative/test_llama3_3b_generation.py::test_llama3_3b_generation
```

Model config: `meta-llama/Llama-3.2-3B`, `max_model_len=128`, `max_num_seqs=1`, single device.

## Call Chain

1. vLLM `_precompile_backbone()` → dummy forward pass
2. torch dynamo → XLA → tt-mlir → tt-metal
3. tt-metal JIT compiles `sdpa_flash_decode` kernel
4. trisc1 firmware link fails on BH exponential LLK

## Key Files

- **Failing LLK source**: `tt_metal/third_party/tt_llk/tt_llk_blackhole/common/inc/sfpu/ckernel_sfpu_exp.h:143`
- **SFPI header**: `/opt/tenstorrent/sfpi/include/sfpi.h:766`
- **Toolchain**: `/opt/tenstorrent/sfpi/compiler/bin/riscv-tt-elf-g++` (GCC 15.1.0)
- **Linker script**: `blackhole/kernel_trisc1.ld`
- **Log**: `test_llama3_3b_generation_base.log`

## Questions for tt-metal Team

1. Is this a known issue with the BH `sdpa_flash_decode` kernel?
2. Is there a toolchain or tt-metal version that fixes the BH exponential LLK register pressure?
3. Is there a way to force a non-flash-decode SDPA path as a workaround?

## Next Steps

- [ ] Share summary with tt-metal team and check if known
- [ ] Check if newer SFPI toolchain or tt-metal version resolves it
- [ ] Investigate workaround: alternative SDPA implementation that avoids `sdpa_flash_decode`
- [ ] Check if other models also hit this on BH (any model using scaled dot product attention)
