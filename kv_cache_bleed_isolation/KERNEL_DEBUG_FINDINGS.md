# SDPA Decode Kernel Debug Findings

## Summary

The causal mask in the SDPA decode kernel is generated correctly and IS applied
to the QK scores. The bug is specifically in the `fill_tile_partial` function
in `dataflow_common.hpp`, which fails to correctly mask positions for
`cur_pos_in_tile >= 14`.

## How kernels are edited and tested

The SDPA decode kernel source files are JIT-compiled at runtime from
`TT_METAL_HOME` (`third_party/tt-mlir/src/tt-mlir/third_party/tt-metal/src/tt-metal`).
Edits to these files take effect on the next TTNN op invocation without any rebuild.

The standalone TTNN test (`test_ttnn_sdpa_decode_adjacency.py`) can be run from
the tt-xla root to pick up kernel changes immediately.

## Key experiments

### 1. All-inf mask test

Replaced `generate_mask`'s tile fill logic with `fill_tile(NEG_INF)` for all tiles.

**Result**: Output magnitude dropped from 5.5 to 0.55. Proves the mask IS being
generated in the writer kernel and IS being consumed by the compute kernel.

### 2. Forced cur_pos sweep

Overrode `cur_pos` in the writer's `generate_mask` to different values while
keeping the actual `cur_pos_tensor` at 14 for the compute kernel.

| Writer cur_pos | cur_pos_in_tile | Padding leak | Status |
|---|---|---|---|
| 0 | 0 | 0.0 | FIXED |
| 13 | 13 | 0.0 | FIXED |
| 14 | 14 | 5.22 | LEAK |
| 15 | 15 | 5.14 | LEAK |
| 16 | 16 | 6.89 | LEAK |

The boundary is between `cur_pos_in_tile=13` (mask works) and
`cur_pos_in_tile=14` (mask fails). `fill_tile_partial` produces a correct mask
for positions 0-13 but fails for 14+.

### 3. Fusion path vs explicit path

Disabled `DYNAMIC_CHUNK_SIZE` mask fusion (forced `add_mask_fusion=false`) to
use the explicit `add_block_inplace` path. Same leak (5.22). The bug is in the
mask DATA, not in how it's applied.

### 4. Manual face-layout mask fill

Replaced `fill_tile_partial` with a manual loop that writes -inf to each face
based on the tile column layout. Same leak (5.22). This suggests either:
- The face layout assumption (face 0: cols 0-15, face 1: cols 16-31) is wrong
- Or the compute kernel's matmul uses a different column ordering than the mask

## The bug location

**File**: `ttnn/operations/transformer/sdpa_decode/device/kernels/dataflow/dataflow_common.hpp`
**Function**: `fill_tile_partial` (line 44)

The function is supposed to fill columns `cur_pos_in_tile + 1` through 31 with
`-inf` and leave columns 0 through `cur_pos_in_tile` as zero. It works correctly
for `cur_pos_in_tile <= 13` but fails for `cur_pos_in_tile >= 14`.

The suspicious code is the face boundary calculation:
```cpp
int face_start = (cur_pos_in_tile < 15) ? 0 : 1;
uint32_t fill_pos_in_face = (cur_pos_in_tile + 1) % 16;
```

At `cur_pos_in_tile=14`: `face_start=0`, `fill_pos_in_face=15`. The uint32
loop (`for idx = fill_pos_in_uint32_face; idx < num_cols_in_uint32_face`)
becomes `for idx = 8; idx < 8` which doesn't execute. Only the single
odd-position uint16 write at column 15 runs. This SHOULD be correct (fills
column 15 with -inf, faces 1,3 fill 16-31), but empirically doesn't mask.

## CRITICAL: Kernel fix does NOT fix vLLM bleed

After proving the TTNN standalone test is fixed by `cur_pos -= 1` (max_diff goes
from 5.22 to 0.0), I ran the vLLM bleed test with the same kernel change:
**still 12/20 failures.** Also combined with `apply_mask_at_last_chunk = true`
forced in the compute kernel: still 10/20 failures.

Verified via syntax error injection that the vLLM PJRT path DOES compile from
the same src directory. Also cleared `~/.cache/tt-metal-cache/` to ensure no
stale cached kernels. The kernel changes are definitively being picked up.

**Conclusion: the standalone TTNN test and the vLLM bleed are TWO SEPARATE ISSUES.**

The `fill_tile_partial` bug is real (the standalone test proves it), but it's
not the cause of the vLLM KV cache bleed. The vLLM bleed has a different root
cause, possibly in:
- The PJRT runtime's tensor management (views, copies, buffer aliasing)
- The compiled graph's handling of the kv_cache tensor between prefill and decode
- The vLLM scheduler's slot assignment or block table management
- Something in how `paged_fill_cache` or `paged_update_cache` interacts with
  the actual model execution

## Next steps

1. Go back to debugging the vLLM bleed independently of the TTNN masking bug
2. The TTNN masking bug (`fill_tile_partial` failing at `cur_pos_in_tile >= 14`)
   is still a real bug that should be fixed in tt-metal, but it's NOT the root
   cause of the vLLM KV cache bleed
3. Investigate the PJRT runtime path more carefully — the bleed may be in
   buffer management, not in the SDPA kernel

## Files modified (in src/ tree, JIT-compiled)

- `ttnn/.../sdpa_decode/device/kernels/dataflow/dataflow_common.hpp` — mask generation
- `ttnn/.../sdpa_decode/device/kernels/compute/sdpa_flash_decode.cpp` — mask application
- All changes have been REVERTED to original state
