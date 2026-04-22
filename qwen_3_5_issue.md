# Qwen3.5-VL Vision Encoder — Conv3d L1 Overflow / Bad-PCC Post-Mortem

## Summary

Running `qwen_3_5_vision_encoder_demo.py` under `torch.compile(backend="tt")`
originally **crashed** with an L1 circular-buffer overflow inside
`ttnn.experimental.conv3d` (patch-embedding). Fixing the crash exposed a
deeper issue: the ttnn Conv3d kernel with multi-block C_in requires
`C_in_block` to be a multiple of `TILE_WIDTH=32`, not just the documented
`l1_alignment=16`. For Qwen (post-pad `C_in=32`) **no legal `C_in_block` <
`C_in` exists that satisfies this hidden constraint**, so ttnn.conv3d
fundamentally cannot handle this shape in L1 today.

Evidence for the hidden constraint: tt-metal's own `models/demos/qwen3_vl/tests/test_model.py:141`
bypasses ttnn.conv3d and runs the CPU PyTorch `patch_embed` as a reference
(`# Use ref model for conv3d for now`). Every working ttnn.conv3d config in
tt-metal (e.g. `tt_dit/layers/conv3d.py:22-50`) uses `C_in_block ∈ {128, 256}`
— all multiples of 32.

Two architectural fixes landed in `tt-mlir`; neither restores PCC on its own
for this shape — **Qwen needs a Conv3d → matmul decomposition** (next step,
documented below):

1. The blocking heuristic was moved from a late workarounds pass into the
   TTIR→TTNN lowering site so weight-const-eval and the runtime
   `Conv3dConfigAttr` share a single `cInBlock` value. (Fixes the
   compile-time / runtime layout mismatch that produced PCC ≈ 0.25 silently.)
2. The heuristic was tightened to require `C_in_block % TILE_WIDTH == 0` so
   it never emits a configuration that the kernel will silently miscompute.
   For Qwen this means the heuristic can no longer hide the L1 overflow —
   which is correct behavior until decomposition is in place.

## Shape seen by the Conv3d (post-workaround padding)

Reading the generated IR confirmed the op the kernel actually runs is **not**
the pristine PyTorch shape. Prior workarounds pad the input channel dim from
`C_in = 3` to `C_in = 32` to hit tile alignment, and the kernel ends up
`(2, 16, 16)`:

```mlir
"ttnn.conv3d"(...) <{
  in_channels = 32 : i32, out_channels = 1152 : i32,
  kernel_size = array<i32: 2, 16, 16>,
  input_depth = 2, input_height = 16, input_width = 16,
  conv3d_config = #ttnn.conv3d_config<c_in_block = 16, c_out_block = 32>,
  ...
}> : (tensor<16x2x16x16x32xbf16, ...>,
       tensor<16384x1152xbf16, ...>, ...)
```

So all of the math below is against `C_in = 32`, `C_out = 1152`,
`kernel = (2, 16, 16)` — not the naive pre-pad `(3, 2, 14, 14)` I started with.

## Round 1 — the L1 overflow (original `qwen_debug.log:2316-2353`)

```
TT_THROW @ tt_metal/impl/program/program.cpp:1136
Statically allocated circular buffers on core range [(x=0,y=0) - (x=7,y=7)]
grow to 2239776 B which is beyond max L1 size of 1499136 B
```

### Why it overflowed

From `third_party/tt-mlir/.../conv3d_program_factory.cpp:81-98`, the two biggest
per-core CBs are both scaled by `matmul_K_t`:

```
patch_size   = kD * kH * kW * C_in_block
matmul_K_t   = ceil(patch_size / 32)
matmul_N_t   = ceil(C_out_block / 32)

vol2col_tiled CB = matmul_M_t * matmul_K_t tiles
weight_tiled  CB = matmul_K_t * matmul_N_t tiles
```

With the runtime defaults (no `Conv3dConfig` → `C_in_block` falls back to full
`C_in = 32`, `C_out_block` to full `paddedCOut = 1152`):
- `patch_size = 2·16·16·32 = 16384` → `matmul_K_t = 512`
- `matmul_N_t = 36`
- `weight_tiled = 512 × 36 = 18432 tiles` — absurd
- Dominates the 2.24 MB-per-core overflow.

### Sanity experiment

Reran with `grid_thw_cpu = [[1, 2, 2]]` (¼ the patches). The reported CB total
was **byte-identical** to the 16-patch case — proving `num_patches` (spatial
blocking) contributes nothing to this op. The real lever is `C_in_block`
(equivalently: reducing `matmul_K_t`).

## Round 2 — the silent PCC bug (`passing_qwen_debug.log`)

First round of fix: I wrote `Conv3dConfigRewritePattern` as a late
`TTNNWorkaroundsPass` rewrite that injected
`conv3d_config = #ttnn.conv3d_config<c_in_block = 16, c_out_block = 32>` on the
Conv3dOp. The demo ran end-to-end — but PCC collapsed:

```
PCC: 0.250355  (threshold: 0.99)
```

### Hard constraints I missed (found in `conv3d_device_operation.cpp:148-206`)

- `C_in_block` must be a multiple of `hal::get_l1_alignment()` — **16
  elements on Wormhole** (L1_ALIGNMENT = 16 bytes, and the `%` check is done
  in elements).
- `C_in_block` must evenly divide `C_in`.
- `C_out_block` must be a multiple of `TILE_WIDTH = 32` and divide
  `padded_C_out`.
- `C_in_num_blocks ≤ total_cores`.

For the post-pad Qwen shape, the only legal `C_in_block` values are
**{16, 32}**; for `C_out = 1152`, legal `C_out_block` are
`{32, 64, 96, 128, 192, 288, 384, 576, 1152}`.

### The real correctness issue

`lib/Conversion/TTIRToTTNN/TTIRToTTNN.cpp::reshapeWeightForConv3d` (lines
2136-2208) physically **pre-blocks** the Conv3d weight at compile time into a
2D layout of shape

```
(num_C_in_blocks × kD × kH × kW × C_in_block,  C_out)
```

The TTIR lowering was calling it with a hardcoded `cInBlock = TILE_WIDTH = 32`.
When my late workarounds pass then *overrode* the runtime `C_in_block` to 16,
the kernel's row interpretation diverged from the const-eval's row ordering.
Concretely, for `C_in = 32`:

- Const-eval (cInBlock = 32, 1 block) stored element `(kD=0, kH=0, kW=0, c=17)`
  at row **17**.
- Kernel (cInBlock = 16, 2 blocks) read row 17 as `(b=0, kD=0, kH=0, kW=1, cb=1)`
  → expected element at global `C_in` index 1, not 17.

Every weight read was off. Dimensions still matched → **no crash**, just
garbage numerics → PCC ≈ random correlation.

### Key lesson

> The Conv3d weight is pre-blocked at the TTIR→TTNN lowering site. Any later
> pass that changes `C_in_block` in the runtime `Conv3dConfig` **must** also
> rewrite the weight layout, or silently return garbage. Easier: pick the
> block sizes at the lowering site so both sides use the same value by
> construction.

## Round 3 — the hidden `TILE_WIDTH` constraint (`new_passing_qwen_debug.log`)

With the round-2 fix in place, the generated IR correctly carried a matching
config **and** pre-blocked the weight consistently:

```mlir
"ttnn.conv3d"(...) <{
  conv3d_config = #ttnn.conv3d_config<c_out_block = 32, c_in_block = 16>, ...
}>
```

Weight const-eval now included the expected 6D reshape + permute
`(3, 0, 1, 2, 4, 5)` to hoist `num_blocks` outermost. I verified by hand that
row-by-row the const-eval output matches the kernel's `(b, kD, kH, kW, cb)`
read pattern — the layout *is* correct logically.

PCC came back at **0.2659** — essentially unchanged from round 2.

### Where the math still breaks

The intermediate 6D tensor (`#ttnn_layout7` / `#ttnn_layout8` in
`new_passing_qwen_debug.log:2061, 2062`) has affine map

```
(d0, d1, d2, d3, d4, d5) -> (d0 * 16384 + d1 * 8192 + d2 * 512 + d3 * 32 + d4, d5)
```

over shape `(2, 2, 16, 16, 16, 1152)` backed by `memref<1024x36x!ttcore.tile<32x32, bf16>>`.
The innermost block dim `d4 = cb = 16` has stride 1 — but the next dim `d3 =
kW = 16` has stride **32**. The logical element count is `16384 × 1152`, but
the tile-backed storage is `32768 × 1152` — **double**. Every tile row has
only 16 valid `cb` values followed by 16 rows of storage padding.

When the subsequent reshape collapses 6D → 2D, ttnn needs to compact-copy
that padded intermediate. Empirically, either that compaction doesn't happen
correctly for this shape, or the kernel ends up reading the padding. Either
way, the output is scrambled.

### Corroborating evidence

- tt-metal's own Qwen3-VL model test
  (`third_party/tt-mlir/.../models/demos/qwen3_vl/tests/test_model.py:141`)
  skips ttnn.conv3d:
  ```python
  patch_input = reference_model.patch_embed(pt_pixel_values)  # Use ref model for conv3d for now
  ```
- Every shape in `tt_dit/tests/unit/test_conv3d.py` uses
  `C_in ∈ {768, 512, 256, 128}` and the production blocking uses `C_in_block ∈
  {128, 256}`, all **multiples of 32**. Tile-aligned cb dim — no padding
  issue.

### Consequence for the heuristic

For Qwen (`C_in = 32`), the only divisor of 32 that is a multiple of 32 is
32 itself (no blocking). No legal `C_in_block < C_in` exists that satisfies
both the device-op divisibility assertion *and* the hidden tile-alignment
requirement. The updated heuristic therefore returns "no blocking" — which
surfaces the L1 overflow again, rather than silently returning garbage. The
correct long-term fix for Qwen is to sidestep ttnn.conv3d entirely (see
follow-ups).

## Final fix

Moved Conv3d-blocking selection into the TTIR→TTNN lowering site and shared
the heuristic with a late safety-net workarounds pattern.

### New / changed files (mirrored in both `/localdev/hshah/tt-mlir` and `third_party/tt-mlir/src/tt-mlir`)

- **New** `include/ttmlir/Dialect/TTNN/Utils/Conv3dBlocking.h` — declares
  `struct Conv3dBlocking { std::optional<uint32_t> cInBlock, cOutBlock; };`
  and `Conv3dBlocking chooseConv3dBlocking(cIn, cOut, kD, kH, kW);`.
- **New** `lib/Dialect/TTNN/Utils/Conv3dBlocking.cpp` — implementation:
  - `cInBlock`: largest multiple-of-16 divisor of `cIn` that keeps
    `kD·kH·kW · cInBlock ≤ 256 tiles × 32 elts` (target `matmul_K_t ≤ 256`).
  - `cOutBlock`: largest multiple-of-32 divisor of `paddedCOut` that keeps
    `matmul_K_t × matmul_N_t ≤ 256` (≈512 KB weight_tiled budget).
  - Returns `{nullopt, nullopt}` when defaults already fit (no regression for
    small Conv3d ops).
- **Modified** `lib/Conversion/TTIRToTTNN/TTIRToTTNN.cpp`
  (`Conv3dOpConversionPattern`, ~line 2037):
  - Calls `chooseConv3dBlocking` with the post-padding `paddedInChannels`.
  - Feeds the chosen `cInBlock` into `reshapeWeightForConv3d` (controls weight
    layout).
  - Builds a matching `Conv3dConfigAttr` and passes it to the emitted
    `Conv3dOp` (controls runtime kernel reads).
- **Modified** `lib/Dialect/TTNN/Transforms/Workarounds/Decomposition/Conv3dConfigRewritePattern.cpp`:
  - Now only sets `c_out_block` (which does not affect weight row ordering).
  - Explicitly refuses to touch `c_in_block` post-lowering — comments reference
    the coupling.
  - Acts as a safety net for direct-TTNN ops that bypass the lowering.
- **Modified** `lib/Dialect/TTNN/Utils/CMakeLists.txt` — add `Conv3dBlocking.cpp`.

### Expected outcome for the Qwen shape

`chooseConv3dBlocking(cIn=32, cOut=1152, kD=2, kH=16, kW=16)`:

- `targetPatchSize = 256·32 = 8192`;
  `maxCInBlock = max(16, 8192/512) = 16`.
- Largest mult-of-16 divisor of 32 that is `< 32`: **16**.
- `effectiveKTiles = ⌈2·16·16·16 / 32⌉ = 256`.
- `maxNTiles = 256/256 = 1` → `maxCOutBlock = 32`; 1152 % 32 = 0 → **32**.

Per-core CB footprint (bf16 tile = 2 KB, fp32 tile = 4 KB):

| CB                   | Tiles | Bytes   |
| -------------------- | ----- | ------- |
| `vol2col_tiled`      | 256   | 512 KB  |
| `weight_tiled`       | 256   | 512 KB  |
| `vol2col_rm`         | 1 pg  | ~16 KB  |
| `matmul_interm` fp32 | 1     | 4 KB    |
| `matmul_result_rm`   | 1     | 2 KB    |
| `cb_reduction_tiled` | 1     | 4 KB    |
| `cb_worker_ack_back` | 1     | 2 KB    |
| `cb_bias_tiled`      | 1     | 2 KB    |
| `cb_zero_tiled`      | 1     | 2 KB    |
| **Total**            |       | **~1.06 MB** |

Budget 1.43 MB → **~370 KB headroom**.

### Sanity against other Conv3d shapes

- Existing graph tests (`tests/torch/graphs/test_conv3d.py`) use
  `in_channels ∈ {12, 32}`, `out_channels ∈ {12, 32, 34}`, `kernel = 3`.
  For all of these, `defaultKTiles × defaultNTiles ≤ 256`, so the helper
  returns `{nullopt, nullopt}` and lowering behaves exactly as before. No
  regression risk.
- DIT-style `(C_in = 768, kernel = 3×3×3, C_out = 1152)`:
  `cInBlock = 256`, `cOutBlock = 32` — roughly matches `tt_dit/layers/conv3d.py`
  hand-tuned values.

## How to verify end-to-end

1. Rebuild tt-mlir and the PJRT plugin against the updated sources.
2. Run:
   ```
   source venv/activate
   python qwen_3_5_vision_encoder_demo.py
   ```
3. Inspect the compiled TTNN IR (`TTXLA_LOGGER_LEVEL=DEBUG` dumps it). Expect:
   - `conv3d_config = #ttnn.conv3d_config<c_in_block = 16, c_out_block = 32>`.
   - `main_const_eval_0` for the weight has an extra 6D reshape + permute
     (from `reshapeWeightForConv3d`'s `numCInBlocks > 1` branch) that did **not**
     appear in the round-1 IR.
4. No `TT_THROW` from `program.cpp:1136`.
5. `PCC ≥ 0.99` at the end of the demo.

## Follow-ups

### P0 — Decompose non-overlapping Conv3d to `reshape → matmul → reshape`

Qwen's patch-embed Conv3d has `kernel == stride == (2, 16, 16)`, i.e. every
input position is covered by exactly one kernel instance. The op is
equivalent to a matmul:

```
input  : (N, T_in, H_in, W_in, C_in)                    # NDHWC
reshape: (N * T_out * H_out * W_out,  kT * kH * kW * C_in)
matmul : weight (kT * kH * kW * C_in, C_out)
reshape: (N, T_out, H_out, W_out, C_out)
```

This sidesteps ttnn.conv3d entirely and uses the well-tuned matmul path,
which has no tile-alignment issues on the K dim and comfortably fits in L1.

Home: `lib/Dialect/TTNN/Transforms/Decomposition/` — add a pattern that
matches Conv3dOp with `stride == kernel_size` and `padding == 0`, emits
reshape + linear (or reshape + matmul + add-bias) in its place, and runs
before the workarounds pass so the op never reaches the kernel.

### P1 — File a tt-metal bug for the hidden `C_in_block` constraint

Either the device-op's assertion should be tightened to `% TILE_WIDTH == 0`,
or the weight preparation path should be fixed to compact the tile-padded
6D intermediate. Today the kernel silently miscomputes whenever
`C_in_block < TILE_WIDTH`, which is a correctness landmine. Attach the Qwen
repro.

### P2 — CI coverage

- **Lit test** under `test/ttmlir/Dialect/TTNN/` piping a large-`K` Conv3d
  (e.g. `C_in=256, kernel=(3,3,3), C_out=1152`) through the TTIRToTTNN
  pattern and `FileCheck`ing both the `conv3d_config` on the op and the 6D
  block-aware reshape+permute in the weight const-eval subgraph.
- **Graph test** mirroring `tests/torch/graphs/test_conv3d.py` with a
  tt_dit-style shape that DOES exercise multi-block C_in (`C_in=256`, `C_in_block=128`)
  so this exact correctness class is caught in CI.

### P3 — Audit the "defaults fit" branch

When `chooseConv3dBlocking` returns `{nullopt, nullopt}`, TTIRToTTNN still
calls `reshapeWeightForConv3d(..., /*cInBlock=*/TILE_WIDTH)`. For ops where
`C_in > TILE_WIDTH` and defaults fit, the runtime uses full `C_in` as its
block size but the weight is pre-blocked with `C_in_block = TILE_WIDTH` —
the same mismatch class we fixed for the explicit-blocking path. Should be
straightforward to thread the chosen cInBlock (including the fallback) into
that call consistently.

### P4 — Drop the C_in → 32 input pad for Conv3d specifically

With raw `C_in = 3`, `patch_size = 2·16·16·3 = 1536`, `matmul_K_t = 48`.
That fits in L1 without any blocking at all, obviating this entire class of
issues for Qwen. Blocked today because the kernel's reader expects
`C_in_block_bytes` to be NoC-aligned (`C_in_block = 3` elements × 2 bytes =
6 bytes is not). Larger change — needs reader kernel updates or a narrow
workaround in tt-mlir that sends the unpadded tensor and relies on the
kernel's ability to handle unaligned reads. Worth considering long-term.

## Files referenced

- `qwen_3_5_vision_encoder_demo.py` — repro
- `qwen_debug.log:2316-2353` — original L1 overflow trace
- `new_qwen_debug.log:2135, 2291` — round-1 IR + still-failing overflow
- `passing_qwen_debug.log:40157` — round-2 PCC failure
- `third_party/tt-mlir/src/tt-mlir/include/ttmlir/Dialect/TTNN/Utils/Conv3dBlocking.h` — new helper
- `third_party/tt-mlir/src/tt-mlir/lib/Dialect/TTNN/Utils/Conv3dBlocking.cpp` — new helper impl
- `third_party/tt-mlir/src/tt-mlir/lib/Conversion/TTIRToTTNN/TTIRToTTNN.cpp` (`Conv3dOpConversionPattern`, ~line 2033) — where blocking is chosen and Conv3dConfigAttr attached
- `third_party/tt-mlir/src/tt-mlir/lib/Dialect/TTNN/Transforms/Workarounds/Decomposition/Conv3dConfigRewritePattern.cpp` — safety-net pattern
- `third_party/tt-mlir/src/tt-mlir/include/ttmlir/Dialect/TTNN/IR/TTNNOpsAttrs.td:769-787` — `Conv3dConfigAttr` definition
- `third_party/tt-mlir/.../ttnn/operations/experimental/conv3d/device/conv3d_device_operation.cpp:148-206` — C_in_block / C_out_block alignment and divisibility asserts
- `third_party/tt-mlir/.../ttnn/operations/experimental/conv3d/device/conv3d_program_factory.cpp:81-226` — CB sizing formulas
- `third_party/tt-mlir/.../models/tt_dit/layers/conv3d.py:22-50` — production config reference
- `tests/torch/graphs/test_conv3d.py` — existing coverage to extend
