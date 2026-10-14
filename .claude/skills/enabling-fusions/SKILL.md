---
name: enabling-fusions
description: Use when turning a missed-fusion candidate into a real tt-mlir MLIR-level fusing pattern — implementing (or un-gating/fixing) a TTIR or TTNN OpRewritePattern so the compiler fuses ops it currently leaves separate. Pairs with the finding-missed-fusions skill, which produces the fusion_todo.yml this skill consumes. Targets tt-mlir MLIR passes only (not the Torch FX path).
allowed-tools: Bash Read Grep Glob Write Edit Task
---

# Enabling Fusions (tt-mlir MLIR level)

## Overview

This skill takes a missed-fusion candidate (from `finding-missed-fusions`' `fusion_todo.yml`) and **enables the fusion in tt-mlir** by adding, un-gating, or fixing an MLIR pattern rewrite. It is the "enable them if easy/possible" counterpart to the detection skill.

Scope is deliberately narrow: **tt-mlir MLIR-level fusing only** — the `ttir-fusing` and `ttnn-fusing` passes. The Torch FX `FusionProvider` / composite path in tt-xla is explicitly out of scope.

The deliverable for a successful run is: pattern code (new or modified) + a FileCheck lit test + a built `ttmlir-opt` that passes that test, plus a short summary. **If the candidate is not cheaply doable in MLIR (e.g. it needs a brand-new tt-metal kernel), record why and stop** — don't force it.

## When to Use

- User asks to "enable this fusion", "implement the missed fusion", "fix the fusion that isn't firing", or hands you a `fusion_todo.yml`.
- After running `finding-missed-fusions`, to act on the highest-priority entries.

When NOT to use:
- The fusion belongs to the Torch FX / composite path (use `docs/source/fusing_and_composite_ops.md` instead).
- The candidate's `source` is `torch`/`triton`/`cuda` with no existing TTNN/TTIR target op — that is a kernel feature request, not an MLIR fusing pattern. Record it and stop.

## Inputs

- A `fusion_todo.yml` (whole file or a single `missed_fusions` entry). Each entry already carries the fields this skill branches on:
  - `dialect_level` — `ttir` or `ttnn` (which pass to extend).
  - `pass_status` — `no_pass`, `pass_exists_not_fired`, or `gated_off`.
  - `owning_pass` / `existing_pattern` — present when a catalog pattern already targets this `fused_op`.
  - `fused_op`, `component_ops`, and per-instance `ttir_segment` / `ttnn_segment` (verbatim IR — reuse directly as FileCheck test input).
- The tt-mlir source tree at `third_party/tt-mlir/src/tt-mlir/` (pinned commit in `third_party/CMakeLists.txt`).
- The fusing catalog at [`../finding-missed-fusions/references/ttmlir-fusing-catalog.md`](../finding-missed-fusions/references/ttmlir-fusing-catalog.md) — the same catalog the detection skill uses. Read it first.

## Process

### 1. Triage by `pass_status`

Branch on `pass_status` before writing any code:

| `pass_status` | What it means | Action |
|---------------|---------------|--------|
| `gated_off` | A catalog pattern (`existing_pattern`) exists but is behind a disabled build/flag | **Easiest.** Re-run with the gating flag on and confirm it fires. No new code. See step 2a. |
| `pass_exists_not_fired` | A catalog pattern targets this `fused_op`, but it didn't match this IR | Reproduce, find why the match guard rejected it, and relax/extend the existing pattern. See step 2b. |
| `no_pass` | No catalog pattern targets this `fused_op` | Scaffold a new `OpRewritePattern`. See step 3. |

Always confirm the candidate isn't already handled by the composite path (catalog "Composite path" section) before doing anything.

### 2a. `gated_off`: flip the flag and verify

The op-model-validated TTNN patterns (SDPA, NLP head ops, split-QKV, TTNN RoPE) need a `TTMLIR_ENABLE_OPMODEL` build plus pass options. Verify the pattern fires once enabled rather than writing code:

```bash
cd third_party/tt-mlir/src/tt-mlir
# TTNN-level, op-model-gated pattern (e.g. SDPAFusing):
build/bin/ttmlir-opt \
  --ttir-to-ttnn-backend-pipeline="enable-fusing-pass=true" \
  --mlir-print-local-scope <repro>.mlir | FileCheck <repro>.mlir
# Or run the standalone pass with explicit options:
build/bin/ttmlir-opt --ttnn-fusing="enable-op-constraints=true" <repro.ttnn>.mlir
```

For TTIR gated patterns use the matching options (`ttir-fusing="ttnn-enable-conv2d-with-multiply-pattern=true"` or `enable-permute-matmul-fusion=true`). If it fires, the "fix" is a configuration/pipeline-default change — document which flag is needed and where it should be enabled. If it still doesn't fire, reclassify as `pass_exists_not_fired` and go to 2b.

### 2b. `pass_exists_not_fired`: fix the existing pattern

1. Build a minimal repro `.mlir` from the entry's `ttir_segment` / `ttnn_segment` (verbatim).
2. Run the owning pass on it (see commands in step 5) and confirm the fusion does *not* happen.
3. Open the `existing_pattern` class in the owning pass's source (catalog gives the file) and read its `matchAndRewrite` / match guards. Common rejection causes: `hasOneUse()` checks, rank/shape constraints, dtype mismatch, an unexpected intervening op (typecast/reshape) the matcher doesn't look through.
4. Relax/extend the guard *narrowly* so it accepts this case without over-matching. Add a `lookThrough<...>` for benign intervening ops where the existing code already uses that helper (see `SDPAFusingPattern.cpp`).
5. Add a positive FileCheck case (step 4) for the newly-covered shape and keep all existing tests green.

### 3. `no_pass`: scaffold a new pattern

1. **Pick the level** using `dialect_level` and the catalog rule of thumb:
   - **TTIR** (`lib/Dialect/TTIR/Transforms/TTIRFusing.cpp`) for framework-shape fusions (matmul+bias→linear, norms, activations, reductions, RoPE, topk, softmax).
   - **TTNN** (`lib/Dialect/TTNN/Transforms/TTNNFusing.cpp`) for hardware-op fusions that depend on TTNN ops/layouts/op-model constraints (conv/matmul+activation, SDPA, NLP head ops, split-QKV).
2. **Choose the anchor op** — the op that terminates the pattern (the one whose result feeds the rest of the graph). Examples from the repo: `AddOp` for conv+bias / matmul+bias, `MultiplyOp` for RMSNorm/SiLU, `DivOp` for softmax, `ReshapeOp` for concat-heads, `MatmulOp` for SDPA. The `OpRewritePattern<AnchorOp>` matches on this type.
3. **Write the pattern class.** Mirror an existing pattern of similar complexity:
   - Simple, single-file pattern → add the class directly in the pass `.cpp` next to siblings (see `ConvAddBias`, `TTNNMatmulAndLinearWithActivation`).
   - Non-trivial pattern (helpers, multi-op walk, op-model validation) → put it in its own `lib/Dialect/TT{IR,NN}/Transforms/Fusing/<Name>Pattern.cpp` with a header under `include/ttmlir/Dialect/TT{IR,NN}/Transforms/Fusing/<Name>Pattern.h` (see `SDPAFusingPattern`, `RoPEFusingPattern`, `SplitQKVFusingPatterns`).
   - Structure: `matchAndRewrite(AnchorOp srcOp, PatternRewriter &rewriter)` → validate guards (return `failure()` early), build the fused op via `rewriter.create<FusedOp>(...)`, then `rewriter.replaceOp(srcOp, ...)`. Let DCE remove the now-dead component ops.
   - For TTNN ops with hardware constraints, validate the candidate fused op before committing, exactly as `SDPAFusing`/`NLPConcatHeadsDecodeFusing` do (`IsolatedIRValidationWrapper` / `op_constraint_validation::validateOperation`), and guard the registration with `#ifdef TTMLIR_ENABLE_OPMODEL` + `enableOpConstraints`.
4. **Register the pattern** in the owning pass's `runOnOperation()`:
   - TTIR: add `patterns.add<MyFusionPattern>(&getContext());` in `TTIRFusing.cpp` (alongside the existing `patterns.add<...>` block). If it should be optional, add a pass `Option<...>` in `include/ttmlir/Dialect/TTIR/Transforms/Passes.td` and guard the `add` with it.
   - TTNN: add `patterns.add<MyFusing>(&getContext());` (or `(&getContext(), validationConfig)` for op-model patterns) in `TTNNFusing.cpp`, inside the `#ifdef TTMLIR_ENABLE_OPMODEL` / `enableOpConstraints` block if it needs validation.
   - Add the new `.cpp` to the dialect's transforms `CMakeLists.txt` if you created a new file.

### 4. Add a FileCheck lit test

Create `test/ttmlir/Dialect/TT{IR,NN}/fusing/<name>.mlir` following the existing tests in those directories:

```mlir
// RUN: ttmlir-opt -ttir-fusing -o %t %s
// RUN: FileCheck %s --input-file=%t

module {
  func.func @my_case(%arg0: ..., ...) -> ... {
    // CHECK-LABEL: func.func @my_case
    // CHECK: "ttir.<fused_op>"
    // CHECK-NOT: "ttir.<component_op_a>"
    // CHECK-NOT: "ttir.<component_op_b>"
    <verbatim component IR from the fusion_todo.yml ttir_segment/ttnn_segment>
    return ...
  }
}
```

- Use the entry's verbatim `ttir_segment` / `ttnn_segment` as the function body so the test reflects the real model shape.
- TTNN-level tests typically run through the backend pipeline: `// RUN: ttmlir-opt --ttir-to-ttnn-backend-pipeline="enable-fusing-pass=true" ...` (see `test/ttmlir/Dialect/TTNN/fusing/resnet_pattern_fusing.mlir`). Add `enable-fusing-conv2d-with-multiply-pattern=true` or other options as the pattern requires.
- Include at least one **negative** case (a near-miss that must NOT fuse) when the match guards are non-trivial — see `matmul_with_bias_negative.mlir`, `split_query_key_values_and_split_heads_negative.mlir`.

### 5. Build and verify

```bash
cd third_party/tt-mlir/src/tt-mlir
# Build just the opt tool (fast iteration):
cmake --build build --target ttmlir-opt

# Run your new pattern directly on the repro:
build/bin/ttmlir-opt -ttir-fusing -o /tmp/out.mlir /tmp/repro.mlir   # TTIR
build/bin/ttmlir-opt --ttir-to-ttnn-backend-pipeline="enable-fusing-pass=true" \
  -o /tmp/out.mlir /tmp/repro.mlir                                    # TTNN pipeline
FileCheck test/ttmlir/Dialect/TTIR/fusing/<name>.mlir --input-file=/tmp/out.mlir

# Run the lit test suite (or just the fusing dir):
cmake --build build --target check-ttmlir
# Targeted: llvm-lit -v build/test/ttmlir/Dialect/TTIR/fusing/<name>.mlir
```

Op-model-gated TTNN patterns require the build to be configured with `TTMLIR_ENABLE_OPMODEL`; if the current `build/` was not, note that the verification needs an op-model-enabled build.

### 6. Confirm with the detection skill

Regenerate fresh IR for the affected model (or re-dump the repro through the full pipeline) and re-run `finding-missed-fusions`. The fixed candidate should no longer appear in `fusion_todo.yml` — that silence is the success signal, end to end.

## Deliverables

For each enabled fusion, report:
- The `fused_op`, `dialect_level`, and which path was taken (`gated_off` flag flip / `pass_exists_not_fired` fix / `no_pass` new pattern).
- Files touched: pattern `.cpp`/`.h`, pass registration, `CMakeLists.txt`, and the FileCheck test path.
- Number of report `instances` this covers.
- The exact `ttmlir-opt` + `FileCheck` command used to verify, and whether an op-model build was needed.

If a candidate could not be enabled cheaply (needs a new tt-metal kernel, or the guard relaxation would over-match), say so explicitly and leave the entry for kernel/feature work — do not write a speculative pattern.

## Common Mistakes

- **Writing a new pattern when one already exists.** Always honor `pass_status`: `gated_off` and `pass_exists_not_fired` must reuse `existing_pattern`, not duplicate it.
- **Choosing the wrong dialect level.** Hardware-op fusions (SDPA, conv/matmul+activation, NLP head ops) belong at TTNN; framework-shape fusions belong at TTIR. The catalog's rule of thumb is authoritative.
- **Over-matching.** Relaxing a guard so it fuses cases the hardware op can't handle produces silent miscompiles. Add a negative FileCheck case to pin the boundary.
- **Skipping op-model validation on TTNN patterns.** Constraint-sensitive TTNN ops must be validated before commit (as `SDPAFusing` does) and gated under `TTMLIR_ENABLE_OPMODEL`.
- **Paraphrasing the test IR.** Use the verbatim `ttir_segment` / `ttnn_segment` from the report so the test reflects a real model shape.
- **Forcing a theoretical fusion.** A `source: torch|triton|cuda` candidate with no existing target op is a kernel request — record and stop, don't invent a pattern.
- **Forgetting the CMake entry.** A new `Fusing/<Name>Pattern.cpp` won't compile until it's added to the dialect transforms `CMakeLists.txt`.
