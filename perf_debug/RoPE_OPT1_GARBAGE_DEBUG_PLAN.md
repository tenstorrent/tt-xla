# RoPE Garbage Output at opt_level=1 — ROOT CAUSE FOUND AND FIX VERIFIED

## TL;DR

The garbage output you've been seeing at `opt_level=1` with `e60ba14a8` is
caused by `RoPEExpandedFusing` in tt-mlir falsely matching the
`TTRotaryEmbedding` pattern. The fused `ttnn.rotary_embedding` kernel
produces wrong output for the K tensor (8 KV heads), corrupting attention,
yielding incoherent tokens. **A 14-line patch in `RoPEFusingPattern.cpp`
fixes the issue.** Build complete, two test runs (one with DEBUG IR
capture) verified clean coherent output and 0 `ttnn.rotary_embedding` ops
in IR (i.e., fusion is correctly blocked).

## Empirical evidence (before vs after fix)

**Llama-3.2-1B `test_output_coherence_nongreedy` at opt_level=1:**

| | Output |
|---|---|
| **Before fix (garbage)** | `' A Fox and his friends was sitting in a village. The fox and the river. He said: "What a river.\nFox. MFCO River.\nFox. Fox, fox: How did you river.\nThey.\nI remember, said. Fox, fox. Fox.\nHow river. Fox. I: Fox'` |
| **After fix run 1** | `' —Robert Frost\nPoetry Friday – February 11, 2022: The Riddle of the Fox\nFebruary 11, 2022 in Poems to Share\nRead the poem. Then tell us what you think. What are the riddles the fox poses for the reader? What do those riddles'` |
| **After fix run 2** | `' A fox and a cat. A fox and a lion. A fox and a bear. One fox and a real bear. A fox and a real lion. A fox and a queen.\nWhat was the first animal that I remember? ...'` |

**Llama-3.2-3B `test_llama3_3b_generation_opt_level_1[batch1]` (canonical original failing case):**

| | Output |
|---|---|
| **Before fix (intermittent garbage)** | `' A young man came to town there with his  to tell me I 'Dayana and I met a long in  and said: my his dear ana am on the cafe - a person, "In a cow: a million, "I have a can't you a month, "people "I want to"'` |
| **After fix** | `' Well, anyone can tell you a story. But the ability to craft a story and communicate it to a large audience is a skill that not everyone has.'` |

All post-fix runs produce coherent, on-topic English. IR confirms the
fix is taking effect — **0 `ttnn.rotary_embedding` ops generated**
(fusion blocked, falls back to the unfused expanded-form computation
that the hardware correctly executes via plain elementwise ops).

## Root cause chain

1. **`e60ba14a8` introduced `TTRotaryEmbedding`** to `overrides.py`,
   applied **unconditionally** to ALL Llama RoPE models (not gated on
   `enable_trace`). Even at `enable_trace=False` it replaces vLLM's
   native `RotaryEmbedding`.

2. **`TTRotaryEmbedding` produces the expanded-form RoPE pattern**:
   it computes `cos = freqs.cos()`, `sin = freqs.sin()` (independent
   ops), then `ApplyRotaryEmb.forward_static` (NEOX-style) emits:
   ```
   o1 = x1 * cos - x2 * sin
   o2 = x2 * cos + x1 * sin
   output = cat((o1, o2), dim=-1)
   ```
   This matches `matchExpandedRope` in `RoPEFusingPattern.cpp`.

3. **`f3ddbfb6b`'s `isPackedCosSinPair` check returns false** (cos and
   sin are independent ops, not slices of a packed source), so the
   fusion proceeds.

4. **The fusion produces a `ttnn.rotary_embedding` op** with mismatched
   shapes. Confirmed in IR (`/tmp/rope_debug_failing.log`):
   ```mlir
   %23 = "ttnn.rotary_embedding"(%11, %21, %22)
       : (<1x1x8x64>, <1x1x1x64>, <1x1x1x64>) -> <1x1x32x64>
   ```
   For Llama-3.2-1B's K (8 KV heads, decode), input is `<1x1x8x64>`
   but the declared output is `<1x1x32x64>`!

5. **The 8→32 inflation comes from `RotaryEmbeddingOpRewritePattern.cpp`**
   (the seq_len padding workaround). It treats `resultShape[size-2]`
   as `seq_len` and pads to a tile multiple. But TTRotaryEmbedding's
   layout is `[B, S=1, H, D]` — dim -2 is **heads** (8 for K, 32 for
   Q), not seq_len.

6. **The kernel can't broadcast cos/sin's dim -2 from 1**. The example
   in the op definition shows them matching exactly (`<1x32x1024x64>`
   input, `<1x1x1024x64>` cos, both dim -2 = 1024). With cos
   `<1x1x1x64>` and input `<1x1x8x64>` (inflated to `<1x1x32x64>`),
   the kernel reads `cos[1..31]` out-of-bounds → garbage rotation
   for K heads 1-7.

7. **K heads 1-7 corrupt** → wrong attention values → garbage
   output tokens. Q (`<1x1x32x64>`) is unaffected because dim -2 =
   32 is already tile-aligned, the workaround doesn't trigger, and
   the kernel runs without padding.

## Why opt_level=1 specifically

In `tt-mlir/lib/Dialect/TTNN/Pipelines/TTNNPipelines.cpp:200-225`,
`createTTNNFusingPass` only adds the RoPE fusion patterns
(`RoPERotateHalfFusing`, `RoPEExpandedFusing`, `RoPEDecodeFusing`)
when `enableOpConstraints == true`, which is true only when the
optimizer is enabled (opt_level >= 1). At opt_level=0 no RoPE
fusion runs, so the bug doesn't surface.

## The fix (currently staged in tt-mlir submodule)

In `lib/Dialect/TTNN/Transforms/Fusing/RoPEFusingPattern.cpp::matchExpandedRope`,
right after the existing `isPackedCosSinPair` check (added by `f3ddbfb6b`):

```cpp
// The fused RotaryEmbedding kernel requires cos/sin's seq_len (dim -2)
// to match the input's seq_len. Some upstream computations (e.g. on-the-fly
// cos/sin via outer + cos/sin in decode mode) produce cos/sin with dim -2 = 1
// while the input's dim -2 represents heads (>1). The kernel cannot broadcast
// dim -2 from 1, and pairing this with the seq_len padding workaround
// produces silent wrong outputs. Skip fusion when shapes are inconsistent.
auto cosShapeType = mlir::dyn_cast<RankedTensorType>(cosValue.getType());
auto resultShapeType = mlir::dyn_cast<RankedTensorType>(c.concatOp.getType());
if (cosShapeType && resultShapeType &&
    cosShapeType.getRank() >= 2 && resultShapeType.getRank() >= 2) {
  int64_t cosSeqLen = cosShapeType.getShape()[cosShapeType.getRank() - 2];
  int64_t resSeqLen = resultShapeType.getShape()[resultShapeType.getRank() - 2];
  if (cosSeqLen != resSeqLen) {
    return false;
  }
}
```

Path: `/localdev/kmabee/tt-xla/third_party/tt-mlir/src/tt-mlir/lib/Dialect/TTNN/Transforms/Fusing/RoPEFusingPattern.cpp`

## Compatibility check (will not break existing tests)

Existing tt-mlir RoPE tests in `test/ttmlir/Dialect/TTNN/optimizer/ttnn_fusing/rotary_embedding/rope.mlir`:

| Test | Input dim -2 | Cos dim -2 | Match? | Fusion |
|---|---|---|---|---|
| `rope_basic_broadcast` | 1024 | 1024 | ✓ | fires |
| `rope_4d_broadcast` | 128 | 128 | ✓ | fires |
| `rope_expanded_basic` | 1 | 1 | ✓ | fires |
| `rope_expanded_broadcast_b32` | 1 | 1 | ✓ | fires |
| **TTRotaryEmbedding (K/Q)** | **8 or 32** | **1** | ✗ | **blocked** |

Fix is precise — only blocks the buggy case.

## Action items for tomorrow

1. **File a tt-mlir issue** describing this false positive (cousin of #8042/#8054)
2. **Submit a tt-mlir PR** with the patch (analogous to PR #8054)
3. **Once merged, uplift in tt-xla** to roll out the fix to main
4. Until then: keep the patch staged in your local tt-mlir submodule

This is a real bug in **main** of tt-xla — `e60ba14a8` is merged, and ALL
Llama models running through vLLM at opt_level=1 are affected (not just
your branch). The combination of `TTRotaryEmbedding` introduced by
`e60ba14a8` and the existing tt-mlir RoPE fusion produces silent wrong
outputs.
