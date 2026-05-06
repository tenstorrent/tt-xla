---
name: finding-missed-fusions
description: Use when auditing a TTNN model's IR for missed op fusion opportunities — both direct TTNN fusions (a fused ttnn op already exists) and theoretical fusions (the pattern is a single kernel in torch/triton/cuda)
---

# Finding Missed Fusions

## Overview

Op fusion is critical for TTNN model performance. This skill audits a model's TTNN IR for **two classes** of missed fusion:

1. **Direct TTNN fusions** — sequences of ops that have a corresponding fused operation already exposed in the TTNN API.
2. **Theoretical fusions** — patterns that other frameworks (torch, triton, cuda) implement as a single kernel, even if no fused TTNN op exists yet. These surface kernel-work candidates.

The deliverable is a structured YAML report. **If nothing should be fused, produce no output and exit silently** — silence is the success signal.

## When to Use

- Model perf is worse than expected and op fusion is suspected
- Reviewing TTNN IR for a newly-brought-up model
- Validating that a compiler pass produced the fused ops you expected
- User asks to "check fusions", "audit fusions", or "find missed fusions"

When NOT to use:
- Functional correctness debugging (use `superpowers:systematic-debugging`)
- Reviewing StableHLO/HLO before lowering — fusions live at the TTNN level

## Input

The skill operates on **two input files** that the user provides — both IR levels are always required, since TTIR is always available alongside TTNN (it's produced earlier in the same compile):

| Level | What it is | How to extract |
|-------|-----------|----------------|
| **TTIR** | The TTIR-dialect MLIR — closer to the framework, before backend lowering. Fusion passes often live at this level. | `.mlir` file, or section inside a log between `module { ... }` |
| **TTNN** | The TTNN-dialect MLIR — after lowering to backend ops, with concrete layouts and dealloc/typecast scaffolding. Shows the actual runtime cost. | `.mlir` file, or section inside a log |

Either may arrive as its own `.mlir` file or as a section inside an execution log `<model>.log`. Both representations of the *same* missed fusion are valuable: the TTIR side is easier to write a fusion pass against, the TTNN side shows the post-lowering shape and is what the runtime actually executes.

## Process

1. **Identify the input file(s)** the user provided. For each, determine:
   - Is it a standalone `.mlir`, or a log with embedded IR (locate the `module { ... }` block)?
   - Does it contain TTIR (`ttir.*` ops) or TTNN (`ttnn.*` ops)? The dialect prefix is the tell.
2. **Fetch the TTNN op reference**: https://docs.tenstorrent.com/tt-metal/latest/ttnn/_sources/ttnn/api.rst.txt — this is the source of truth for what fused ops exist.
3. **Scan for direct fusions**: walk the IR for op sequences that match a fused TTNN op (e.g., separate `mean → sub → mul → rsqrt → add` collapsing to `ttnn.layer_norm`). Scan TTIR for the same patterns at the higher level.
4. **Scan for theoretical fusions**: identify patterns that map to single kernels in torch/triton/cuda (e.g., `matmul → softmax → matmul` → `scaled_dot_product_attention`).
5. **Cross-reference TTIR ↔ TTNN** when both are available: line up matching `loc(#locNNNN)` references so each `instance` in the report can carry both representations.
6. **Fan out subagents** for parallel scans across large IR dumps — split by line ranges or by op family. Each subagent returns a partial list of `missed_fusions` entries which you merge.
7. **Decide**:
   - Nothing found → return without writing a file. Do nothing.
   - Found candidates → write `fusion_todo.yml` in the schema below.

## Output Schema: `fusion_todo.yml`

The report MUST be valid YAML so it can be consumed by tooling — including MLIR pass authors who will write pattern-matchers against `ttir_segment` / `ttnn_segment`. Use this schema verbatim:

```yaml
input:
  ttir: model.ttir.mlir             # optional path; omit if unavailable
  ttnn: dit-trace.log               # optional path; may be a log containing the IR

missed_fusions:
  - fused_op: ttnn.linear
    source: ttnn                    # one of: ttnn | torch | triton | cuda
    component_ops:
      - ttnn.matmul
      - ttnn.add
    instances:
      - ir_loc: "#loc16303"         # MLIR location ref(s), if present
        ttir_segment: |
          %5 = "ttir.matmul"(%input, %weight_q, %3) <{transpose_b = true}> : (tensor<8190x768xbf16>, tensor<3072x768xbf16>, tensor<8190x3072xbf16>) -> tensor<8190x3072xbf16> loc(#loc19102)
          %7 = "ttir.add"(%5, %bias_q, %6) : (tensor<8190x3072xbf16>, tensor<1x3072xbf16>, tensor<8190x3072xbf16>) -> tensor<8190x3072xbf16> loc(#loc16303)
        ttnn_segment: |
          %1727 = "ttnn.matmul"(%1725, %weight_q) <{transpose_b = true}> : (tensor<8190x768xbf16, #ttnn_layout112>, tensor<3072x768xbf16, #ttnn_layout115>) -> tensor<8190x3072xbf16, #ttnn_layout113> loc(#loc19102)
          %1728 = "ttnn.add"(%1727, %568) <{dtype = #ttcore.supportedDataTypes<bf16>}> : (tensor<8190x3072xbf16, #ttnn_layout113>, tensor<1x3072xbf16, #ttnn_layout20>) -> tensor<8190x3072xbf16, #ttnn_layout113> loc(#loc16303)
        notes: |
          matmul + bias-add — collapses to ttnn.linear with bias operand.
          model location: q-projection inside DiT attention block (layer 14).
          source torch ops: nn.Linear(in=768, out=3072, bias=True) — the bias add
          comes from the Linear's `bias` parameter, lowered as a separate ttnn.add.

  - fused_op: ttnn.layer_norm        # with weight/bias operands
    source: ttnn
    component_ops:
      - ttnn.layer_norm              # currently called with operandSegmentSizes = [1, 0, 0]
      - ttnn.add
      - ttnn.multiply
    instances:
      - ir_loc: "#loc, #loc15643, #loc16305"
        ttir_segment: |
          %14 = "ttir.layer_norm"(%13, %ln_init) <{epsilon = 9.99999997E-7 : f32}> : (tensor<1x8190x3072xf32>, tensor<1x8190x3072xf32>) -> tensor<1x8190x3072xf32> loc(#loc)
          %15 = "ttir.add"(%scale_msa, %one, %15_init) : (tensor<1x1x3072xf32>, tensor<1x8190x3072xf32>, tensor<1x8190x3072xf32>) -> tensor<1x8190x3072xf32> loc(#loc15643)
          %16 = "ttir.add"(%15, %shift_msa, %16_init) : (tensor<1x8190x3072xf32>, tensor<1x1x1xf32>, tensor<1x8190x3072xf32>) -> tensor<1x8190x3072xf32> loc(#loc16305)
          %17 = "ttir.multiply"(%14, %16, %17_init) : (...) -> tensor<1x8190x3072xf32> loc(...)
        ttnn_segment: |
          %1734 = "ttnn.layer_norm"(%1733) <{epsilon = 9.99999997E-7 : f32, operandSegmentSizes = array<i32: 1, 0, 0>}> : (tensor<1x8190x3072xf32, #ttnn_layout104>) -> tensor<1x8190x3072xf32, #ttnn_layout104> loc(#loc)
          %1735 = "ttnn.add"(%531#0, %637) <{dtype = #ttcore.supportedDataTypes<f32>}> : (tensor<1x1x3072xf32, #ttnn_layout29>, tensor<1x8190x3072xf32, #ttnn_layout104>) -> tensor<1x8190x3072xf32, #ttnn_layout104> loc(#loc15643)
          %1736 = "ttnn.add"(%1735, %250) <{dtype = #ttcore.supportedDataTypes<f32>}> : (tensor<1x8190x3072xf32, #ttnn_layout104>, tensor<1x1x1xf32, #ttnn_layout75>) -> tensor<1x8190x3072xf32, #ttnn_layout104> loc(#loc16305)
          %1737 = "ttnn.multiply"(%1734, %1736) <{dtype = #ttcore.supportedDataTypes<f32>}> : (...) -> (...) loc(...)
        notes: |
          unweighted layer_norm followed by affine modulation — adaLN.
          model location: pre-attention norm in each DiT block (post-conditioning modulation).
          source torch ops: nn.LayerNorm(elementwise_affine=False) followed by
          `x * (1 + scale_msa.unsqueeze(1)) + shift_msa.unsqueeze(1)` from the DiT
          AdaLayerNormZero module. If %1735/%1736 reduce to a per-channel scale,
          fold into layer_norm's weight operand instead of leaving as separate ops.

  - fused_op: torch.scaled_dot_product_attention
    source: torch
    component_ops:
      - ttnn.matmul
      - ttnn.multiply
      - ttnn.softmax
      - ttnn.matmul
    instances:
      - ir_loc: "#loc18800"
        ttir_segment: |
          %42 = "ttir.matmul"(%q, %k, %42_init) <{transpose_b = true}> : (tensor<2x24x4096x128xbf16>, tensor<2x24x4096x128xbf16>, tensor<2x24x4096x4096xbf16>) -> tensor<2x24x4096x4096xbf16> loc(#loc18800)
          %43 = "ttir.multiply"(%42, %scale, %43_init) : (tensor<2x24x4096x4096xbf16>, tensor<1xbf16>, tensor<2x24x4096x4096xbf16>) -> tensor<2x24x4096x4096xbf16> loc(#loc18801)
          %44 = "ttir.softmax"(%43, %44_init) <{dimension = -1 : si32}> : (tensor<2x24x4096x4096xbf16>, tensor<2x24x4096x4096xbf16>) -> tensor<2x24x4096x4096xbf16> loc(#loc18802)
          %45 = "ttir.matmul"(%44, %v, %45_init) : (tensor<2x24x4096x4096xbf16>, tensor<2x24x4096x128xbf16>, tensor<2x24x4096x128xbf16>) -> tensor<2x24x4096x128xbf16> loc(#loc18803)
        ttnn_segment: |
          %2001 = "ttnn.matmul"(%q, %k) <{transpose_b = true}> : (...) -> (...) loc(#loc18800)
          %2002 = "ttnn.multiply"(%2001, %scale) : (...) -> (...) loc(#loc18801)
          %2003 = "ttnn.softmax"(%2002) <{dimension = -1}> : (...) -> (...) loc(#loc18802)
          %2004 = "ttnn.matmul"(%2003, %v) : (...) -> (...) loc(#loc18803)
        notes: |
          model location: self-attention QK^T → softmax → AV inside each DiT block.
          source torch ops: F.scaled_dot_product_attention(q, k, v, scale=...) — no
          attention mask, no dropout. Flash-attention-style fused kernel.
          no fused TTNN equivalent yet — kernel feature request.
```

### Field reference

| Field | Required | Description |
|-------|----------|-------------|
| `input.ttir` | yes | Path to the TTIR file/log |
| `input.ttnn` | yes | Path to the TTNN IR file/log |
| `missed_fusions` | yes | Top-level list of fusion candidates |
| `fused_op` | yes | Fully qualified target op (e.g., `ttnn.layer_norm`, `torch.scaled_dot_product_attention`) |
| `source` | yes | `ttnn` (direct fusion) or `torch` / `triton` / `cuda` (theoretical) |
| `component_ops` | yes | Ops that should collapse into `fused_op` |
| `instances` | yes | Every concrete site the pattern was found — count drives prioritization |
| `instances[].ir_loc` | optional | The MLIR `#loc...` ref(s) covering the segment, comma-separated. Use these to align TTIR ↔ TTNN. |
| `instances[].ttir_segment` | yes | **Verbatim** TTIR text from the input file. |
| `instances[].ttnn_segment` | yes | **Verbatim** TTNN IR text from the input file. Drop unrelated `ttnn.deallocate` lines but keep all participating ops. |
| `instances[].notes` | yes | Multi-line block. **Must include**: (a) `model location:` — where in the model this pattern lives (e.g., "pre-attention norm in DiT block"), (b) `source torch ops:` — the original torch/nn ops that lowered to this pattern, to aid manual debugging. Add preconditions / motif names (adaLN, SDPA) as relevant. |

## Common Mistakes

- **Reporting when nothing is missed.** Silence is the success signal. Do not produce an empty `missed_fusions: []` file — produce nothing.
- **Including fusions the compiler already does.** Verify the IR really shows separate ops, not a fused op printed as its components. For example, a `ttnn.layer_norm` with `operandSegmentSizes = [1, 1, 1]` *already* has weight and bias fused — don't flag it.
- **Skipping the theoretical class.** Even when no TTNN fused op exists, surfacing torch/triton/cuda equivalents prioritizes kernel work.
- **Free-form output.** No `- start yaml` markers, no prose-only values, no markdown headings inside the file. The YAML must `yaml.safe_load`.
- **Reporting only one occurrence per pattern.** List *every* instance — the count is what tells the reader whether it's worth fixing.
- **Mixing classes in one entry.** One `fused_op` per entry. If both a direct TTNN fusion and a theoretical fusion apply to overlapping ops, write two entries.
- **Paraphrasing the IR.** Both `ttir_segment` and `ttnn_segment` MUST be copied verbatim from the input file (preserve `%ssa`, attributes, types, `loc(...)`). Pass authors will pattern-match on this text. Reformatting or summarizing it makes the report unusable.
- **Mismatched TTIR / TTNN pairs.** If both segments are provided for an instance, they MUST describe the same logical op sequence — line them up by `loc(#locNNNN)` refs before recording. A mismatched pair is worse than a missing one.
- **Including unrelated `ttnn.deallocate` lines.** Drop deallocates that don't reference the participating SSA values — keep the segment focused on the ops that fuse.
- **Ops not contiguous in the input.** If the pattern's ops are interleaved with unrelated ops, that's still a valid fusion candidate but flag it in `notes` so the pass author knows reordering may be required.
