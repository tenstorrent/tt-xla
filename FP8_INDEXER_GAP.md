# FP8 quantization for the DeepSeek-V3.2 indexer — gap analysis & plan

**Branch:** `hshah/dsa-vllm`
**Date:** 2026-06-14
**Question:** Can the DeepSeek-V3.2 "lightning indexer" FP8 weights run as FP8 on
Tenstorrent hardware (instead of being dequantized to BF16, which negates the
indexer's efficiency)? If the tt-mlir / tt-metal stack supports it, expose it in the
vLLM plugin; if not, document exactly where the gap is and how to bridge it.

## Bottom line

**Not supported anywhere in the stack today — no code was changed.** There is **no
IEEE-FP8 matmul** (and no block-scaled FP8 matmul) in tt-metal or tt-mlir, so the
DeepSeek block-wise `float8_e4m3` indexer weights cannot be consumed on device. The
gap is *foundational* (a missing compute kernel), not a plugin wiring issue, and the
three layers have a strict bottom-up dependency. The findings below are verified
first-hand against the source in this checkout.

---

## What the target actually is

In `vllm/model_executor/models/deepseek_v2.py`, `Indexer.__init__` (lines 635–660),
only **two** of the indexer's layers are FP8-quantized:

| Layer | Shape | Quantized? | Evidence |
|---|---|---|---|
| `wq_b` | 1536 → 8192 | **FP8** | `deepseek_v2.py:635-641` (`quant_config=quant_config`) |
| `wk` | 7168 → 128 | **FP8** | `deepseek_v2.py:642-648` (`quant_config=quant_config`) |
| `weights_proj` | 7168 → 64 | No | `deepseek_v2.py:650-656` (`quant_config=None`) |
| `k_norm` | LayerNorm(128) | No | `deepseek_v2.py:649` |

The FP8 format is DeepSeek **block-wise**:

- dtype `torch.float8_e4m3fn`; a separate `weight_scale_inv` tensor (registered at
  `vllm/.../quantization/fp8.py:379`) with **one scale per 128×128 block**
  (`quant_block_size = 128`, `deepseek_v2.py:660`); scale format `ue8m0`
  (`scale_fmt = "ue8m0"`, line 659).
- On GPU the "lightning" efficiency comes from **three** FP8 primitives:
  1. FP8 `wq_b` / `wk` GEMMs;
  2. FP8 activation-quantized **q·k logits** — `fp8_mqa_logits` via the
     `SparseAttnIndexer` op (`vllm/.../sparse_attn_indexer.py`); this is the dominant
     O(S·L) per-token scoring cost;
  3. an FP8 k-cache (`DeepseekV32IndexerCache`, uint8 value + fp32 block scales,
     `deepseek_v2.py:666-671`).

**What the TT path does today:** all three run in **BF16**. `attention_dsa._indexer_project`
computes the projections in the activation dtype (it explicitly "mirrors
`Indexer.forward` up to but excluding the FP8 quantization"), and `_index_scores` is a
BF16 einsum. The TT reference
(`third_party/tt_forge_models/deepseek/deepseek_v3_2_exp/pytorch/src/modified_model.py`,
`bf16_index`) is BF16-only by design. Running the indexer in BF16 is exactly what
"negates the benefit" — the scorer that is supposed to be cheap is now a small dense
BF16 attention.

---

## The gap, layer by layer (verified)

### Layer 1 — tt-metal (the foundational blocker)

`third_party/tt-mlir/src/tt-mlir/third_party/tt-metal/src/tt-metal`

- The only IEEE FP8 dtype is `DataType::FP8_E4M3 = 8`
  (`tt_metal/api/tt-metalium/experimental/tensor/tensor_types.hpp:29-42`), and it is
  **output-only**. Its in-tree comment: *"narrowly supported — Blackhole only,
  ROW-MAJOR only for now, used exclusively by the DeepSeek V3 prefill combine and
  dispatch ops."* Those are the **MoE token-routing** ops
  (`ttnn/.../operations/experimental/deepseek_prefill/combine/...`), unrelated to the
  indexer or to general matmul.
- Host construction / fills of `FP8_E4M3` **throw**:
  `ttnn/.../operations/creation/creation.cpp:204-207` and `:411-413`,
  `creation_nanobind.cpp:532-534` ("FP8_E4M3 is an output-only dtype … host-side
  construction … is not supported").
- **No matmul consumes FP8.** `grep -rn Fp8_e4m3 ttnn/cpp/ttnn/operations/matmul` →
  empty. **No block-scaled / scaled / mxfp matmul exists** —
  `grep -rniE "block_scaled_matmul|scaled_matmul|mxfp|block_scale|scale_a|scale_b" .../operations/matmul`
  → empty. The matmul signature has no scale-tensor parameter.
- `dequantize` accepts **INT32 input only**
  (`ttnn/.../operations/eltwise/quantization/quantization.cpp`:
  `TT_FATAL(a_dtype == DataType::INT32, "Dequantize only supports int32 inputs for now")`),
  so you cannot even decompose the op as `dequantize(e4m3, scale) → bf16` then matmul.
- `BFLOAT8_B` / `BFLOAT4_B` are tt-metal **block-float** (shared exponent per 32×32
  tile), a *different* format from IEEE E4M3 with arbitrary per-128-block scales.

### Layer 2 — tt-mlir (compiler)

`third_party/tt-mlir/src/tt-mlir` (excluding the nested tt-metal subtree)

- The `TTCore` `DataType` enum
  (`include/ttmlir/Dialect/TTCore/IR/TTCoreOpsEnums.td:10-23`) contains **no IEEE FP8
  type at all** — only `f32`, `f16`, `bf16`, and block-float
  `bfp_f8`/`bfp_bf8`/`bfp_f4`/`bfp_bf4`/`bfp_f2`/`bfp_bf2`, plus ints/bool. The
  compiler cannot even *name* an `e4m3` tensor.
- The StableHLO→TTIR frontend has **no `Float8` / `f8E4M3FN` / `f8E5M2` handling**
  (`grep -rniE "Float8|F8E4M3|F8E5M2" lib/Conversion/StableHLOToTTIR` → empty). It
  accepts `quant.uniform` quantized types, but only with **integer** storage types
  (i8/i32); `ttir.dequantize` / `ttir.quantize` target those, not e4m3 + block scales.
- No block-scaled matmul op in TTIR or TTNN dialects.

### Layer 3 — vLLM plugin (proximate symptom)

- `Fp8LinearMethod.__init__` → `init_fp8_linear_kernel`
  (`vllm/.../quantization/fp8.py:304`) → `choose_scaled_mm_linear_kernel`, which does
  `possible_kernels[current_platform._enum]` against `_POSSIBLE_FP8_KERNELS`
  (`vllm/model_executor/kernels/linear/__init__.py:108`). That dict has keys for
  **CUDA / ROCM / CPU / XPU only**, so on the TT out-of-tree platform it raises
  `KeyError: PlatformEnum.OOT` (~line 229) — at *instantiation*, before any compute.
  This is why the single-layer test sets `hf_overrides={"quantization_config": None}`.
- `vllm_tt/platform.py:173-175` — `supported_quantization` is empty (`"fp8"`
  commented out): the plugin advertises no quantization support.
- The actual GPU block-FP8 compute is `w8a8_block_fp8_matmul` (a Triton kernel); there
  is no TT equivalent, and DeepSeek block-quant does **not** go through the per-tensor
  `_POSSIBLE_FP8_KERNELS` scaled-mm path anyway.
- `experimental_weight_dtype="bfp_bf8"` (`vllm_tt/platform.py:68-69`, passed to PJRT at
  `:145`/`:152`) is a tt-mlir **compile-time block-float** weight conversion — **not**
  IEEE FP8, and orthogonal to vLLM's FP8 infrastructure.

### Strict dependency

```
vLLM plugin (Layer 3)  ──needs──▶  tt-mlir fp8 type+op (Layer 2)  ──needs──▶  tt-metal fp8 matmul (Layer 1)
```

Fixing the plugin's `KeyError` just exposes the missing tt-mlir type; adding the type
just exposes the missing tt-metal kernel. **Nothing runs until tt-metal has an FP8
compute kernel.** The `KeyError` is the symptom; the missing kernel is the disease.

---

## Plan to bridge (true FP8), bottom-up

**1. tt-metal — the hard part (new kernel work; gates everything).**
Add a **block-scaled FP8 matmul**: inputs = `e4m3` weight + `fp32` per-128-block
`weight_scale_inv` (+ optional per-token-group activation scale), BF16/FP32 accumulate.
Concretely:
   - Promote `FP8_E4M3` from output-only to a first-class **input** dtype: host
     construction, `TILE` layout, reader kernels (today blocked by the `creation.cpp`
     throws).
   - Either (a) a fused `dequant(e4m3, block_scale) → bf16` inside the matmul compute
     kernel, or (b) a standalone block-scaled-dequant op feeding the existing matmul.
   - For the indexer **logits** benefit (the dominant cost), separately add an FP8
     activation·activation matmul (the `fp8_mqa_logits` analogue) + the FP8 k-cache
     quant/cache op.

**2. tt-mlir — type + op + lowering.**
   - Add IEEE FP8 element type(s) to `TTCore::DataType`.
   - Teach StableHLO→TTIR to accept `f8E4M3FN` (and either `quant.uniform<f8…>` or a
     dedicated block-scaled-quant attribute carrying `weight_scale_inv`).
   - Add a TTIR + TTNN `block_scaled_matmul` op (or a `dequantize` that takes
     e4m3 + block scales), with a sharding rule for the scale operand, lowering to the
     Layer-1 kernel.

**3. vLLM plugin — expose it (small, once 1 & 2 exist).**
   - Add a TT block-FP8 linear / quant method that keeps `weight` + `weight_scale_inv`
     in FP8 and emits the new TT op. Do **not** route through `_POSSIBLE_FP8_KERNELS`
     (that's per-tensor scaled-mm); DeepSeek is block-quant.
   - Add `"fp8"` to `supported_quantization` and thread `weight_scale_inv` through
     weight loading.
   - In `attention_dsa._indexer_project`, stop dequantizing — call the FP8 `wq_b`/`wk`
     directly. Optionally quantize the k-cache + logits for the full indexer benefit.

**Effort:** realistically a tt-metal kernel + tt-mlir compiler effort measured in
**weeks**, owned outside the plugin. The plugin change is the small last step.

---

## Pragmatic partial option available today (NOT true FP8)

If the near-term goal is just to stop paying BF16 cost for the indexer **weight**
GEMMs, convert `wq_b`/`wk` to tt-metal **block-float `bfp_bf8`** at compile time via the
existing `experimental_weight_dtype` / per-tensor `weight_dtype_overrides` path
(Layer-1/2 support already exists). Caveats, most important first:

1. It's a **different numeric format** (block-float, not e4m3) and requires
   **dequantizing the e4m3 checkpoint to BF16 first**, then recompressing — it does not
   "use" the original FP8 bytes; it recovers on-device 8-bit weight storage/bandwidth.
2. It does **nothing for the dominant O(S²·NH) BF16 score logits** — that's an
   activation·activation einsum, untouched by weight conversion.
3. Needs verification that the weight-convert pass actually catches the indexer
   `ReplicatedLinear` GEMMs.

Treat this as a memory/bandwidth optimization, **not** "FP8 indexer support."

---

## Recommendation

Do not attempt FP8 indexer support in the plugin in isolation — it will dead-end at
the `KeyError` and, even if forced past, at the missing tt-mlir type and tt-metal
kernel. Track it as a stacked, bottom-up effort (tt-metal block-scaled FP8 matmul →
tt-mlir type/op/lowering → plugin quant method). If a quick win is needed before that
lands, evaluate the `bfp_bf8` weight-conversion option for `wq_b`/`wk` with the caveats
above, and keep the BF16 mask-based indexer (already validated — see
`DSA_DECODE_ENABLEMENT.md`) as the functional baseline.

---

## Evidence index (files inspected)

- tt-metal: `tt_metal/api/tt-metalium/experimental/tensor/tensor_types.hpp` (DataType
  enum); `ttnn/.../operations/creation/creation.cpp`, `creation_nanobind.cpp`
  (output-only throws); `ttnn/.../operations/matmul/*` (no scaled/fp8 matmul);
  `ttnn/.../operations/eltwise/quantization/quantization.cpp` (int32-only dequant);
  `ttnn/.../operations/experimental/deepseek_prefill/combine/device/*` (the only
  FP8_E4M3 producer).
- tt-mlir: `include/ttmlir/Dialect/TTCore/IR/TTCoreOpsEnums.td` (DataType enum);
  `lib/Conversion/StableHLOToTTIR/*` (no f8 handling).
- vLLM: `model_executor/kernels/linear/__init__.py` (`_POSSIBLE_FP8_KERNELS`,
  `choose_scaled_mm_linear_kernel`); `model_executor/layers/quantization/fp8.py`
  (`Fp8LinearMethod`, `weight_scale_inv`, block path); `model_executor/models/deepseek_v2.py`
  (`Indexer`); `model_executor/layers/sparse_attn_indexer.py`.
- tt-xla plugin: `integrations/vllm_plugin/vllm_tt/platform.py`
  (`supported_quantization`, `experimental_weight_dtype`);
  `integrations/vllm_plugin/vllm_tt/attention_dsa.py` (`_indexer_project`,
  `_index_scores`).
- TT reference: `third_party/tt_forge_models/deepseek/deepseek_v3_2_exp/pytorch/src/modified_model.py`
  (`bf16_index`, BF16-only indexer).
