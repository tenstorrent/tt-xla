---
name: ONNX Workstreams
overview: Implementation plan for native ONNX support on tt-xla via onnx-mlir bridge and PJRT integration.
parent_plan: onnx_support_options.md
isProject: false
---

# ONNX on TT-XLA — Workstreams & Implementation Plan

**Parent plans:** [onnx_support_options.md](onnx_support_options.md), [onnx_integration_milestones.md](onnx_integration_milestones.md)

This document turns the agreed ONNX strategy into four concrete workstreams. Workstream 1 is the prerequisite for all ONNX frontend work.

---

## Architecture

```mermaid
flowchart LR
  ONNX[".onnx file"] --> Bridge["onnx-mlir-opt\n--convert-onnx-to-stablehlo"]
  Bridge --> SHLO["StableHLO MLIR"]
  SHLO --> Gate["ModuleBuilder\n(direct SHLO path)"]
  Gate --> PJRT["PJRT_Client_Compile\n(format=mlir)"]
  PJRT --> MB["SHLO → TTIR → TTNN → flatbuffer"]
  MB --> Exec["LoadedExecutable.execute()"]
  Exec --> HW["tt-metal device"]
```

**Key insight:** PJRT + `ModuleBuilder` + flatbuffer runtime are framework-agnostic. ONNX integration is a new frontend that produces StableHLO MLIR. No changes to the TTIR/TTNN lowering path are required for the spike.

---

## Workstream 1 — Direct StableHLO ingestion (C++)

**Milestone:** M1.2  
**Status:** In progress  
**Blocks:** All ONNX bridge work

### Problem

`ModuleBuilder::buildModule()` always runs the VHLO deserialize pipeline after parsing MLIR:

1. Parse MLIR (`createVHLOModule`)
2. `convertFromVHLOToSHLO` — runs `createStablehloDeserializePipeline`
3. `runFrontendSHLOPipeline` → tt-mlir SHLO pipeline → TTIR → TTNN

JAX and PyTorch emit **versioned VHLO** from XLA. ONNX bridges (onnx-mlir) emit **raw StableHLO dialect** ops. Step 2 fails on direct StableHLO input.

### Solution

1. **Auto-detect** input format after parse:
   - Module contains `vhlo.*` ops → run VHLO deserialize (existing path)
   - Module contains `stablehlo.*` ops and no `vhlo.*` → skip deserialize
   - Neither (empty/func-only) → default to VHLO path for backward compatibility

2. **Optional compile option** `mlir_input_format`:
   - `auto` (default) — use detection
   - `vhlo` — force VHLO deserialize
   - `stablehlo` — skip deserialize (ONNX / hand-fed SHLO)

### Files touched

| File | Change |
|------|--------|
| `pjrt_implementation/inc/api/compile_options.h` | Add `mlir_input_format` field |
| `pjrt_implementation/src/api/compile_options.cc` | Parse `mlir_input_format` option |
| `pjrt_implementation/inc/api/module_builder/module_builder.h` | Add `MlirInputFormat` enum + `detectMlirInputFormat()` |
| `pjrt_implementation/src/api/module_builder/module_builder.cc` | Conditional skip of VHLO deserialize |
| `pjrt_implementation/src/api/unit_tests/module_builder_tests.cc` | Unit tests for detection |

### Exit criteria

- [ ] Unit tests pass for StableHLO and VHLO detection
- [ ] Existing JAX/PyTorch compile path unchanged (VHLO still deserializes)
- [ ] Hand-fed StableHLO MLIR reaches `runFrontendSHLOPipeline` without error
- [ ] Documented compile option available for debugging

### Validation commands

```bash
# Build
cmake --build build --target TTPJRTTests

# Run unit tests (includes module_builder detection tests)
ctest --test-dir build -R PJRTUnitTests --output-on-failure

# Filter to module builder tests only
./build/tests/pjrt/TTPJRTTests --gtest_filter='ModuleBuilderUnitTests.*'
```

---

## Workstream 2 — Build onnx-mlir pinned to tt-mlir

**Milestone:** M1.3  
**Prerequisite:** Workstream 1 complete

### Goal

Reproducible `onnx-mlir-opt` binary aligned with tt-mlir's LLVM/MLIR commit.

### Steps

1. Read MLIR commit from `third_party/tt-mlir` submodule
2. Add `third_party/onnx-mlir` or `tools/onnx/build_onnx_mlir.sh`
3. Smoke test: 1-op Add ONNX → StableHLO MLIR
4. Document env var `TT_ONNX_MLIR_OPT` pointing to built binary

### Exit criteria

- [ ] `onnx-mlir-opt --convert-onnx-to-stablehlo` runs on trivial Add model
- [ ] Output parses as MLIR with `stablehlo` ops

**Effort:** 1–3 weeks (MLIR pin alignment is the long pole)

---

## Workstream 3 — `tt_onnx` Python package

**Milestone:** M1.5  
**Prerequisite:** Workstreams 1 + 2

### Layout

```
python_package/tt_onnx/
  __init__.py
  bridge.py          # onnx-mlir-opt subprocess wrapper
  compiler.py        # PJRT compile via JAX client
  runtime.py         # buffer I/O + execute
  session.py         # ONNXSession high-level API
tools/onnx_spike/
  compile_add.py
tests/onnx/
  test_add_e2e.py
```

### API sketch

```python
from tt_onnx import ONNXSession

session = ONNXSession("model.onnx", device="tt")
outputs = session.run({"input": np_array})
```

Compile options passed to PJRT:

```python
{
    "export_path": "onnx_artifacts",
    "export_model_name": "resnet18",
    "mlir_input_format": "stablehlo",  # or rely on auto-detect
}
```

### Exit criteria

- [ ] Single command: Add ONNX → compile → execute on TT device
- [ ] Numerical match vs ONNX Runtime CPU reference

**Effort:** 1–2 weeks

---

## Workstream 4 — Tests and incremental validation

**Milestones:** M1.6–M1.11  
**Prerequisite:** Workstream 3

### Order

| Step | Test | Proves |
|------|------|--------|
| 1 | Add ONNX → SHLO dump | Bridge works |
| 2 | Add ONNX → PJRT compile-only | SHLO ingestion + tt-mlir |
| 3 | Add ONNX → device execute | Full stack |
| 4 | 2-layer MLP ONNX | Multi-op graph |
| 5 | Op matrix (~15–20 ops) | Scaling signal |
| 6 | ResNet-18 from `tt_forge_models` | First real model |

Reuse existing infra:

- IR dumps via `export_path` / `export_model_name` (same as `examples/pytorch/export_ir_example.py`)
- Op failures via `tests/op_by_op/` on dumped `shlo_compiler_*.mlir`
- Model loaders from `third_party/tt_forge_models/*/onnx/loader.py`

---

## Timeline (1 engineer, Cursor-assisted)

| Week | Workstream | Deliverable |
|------|------------|-------------|
| 1 | WS1 | Direct SHLO ingestion + unit tests |
| 2 | WS2 + WS3 start | onnx-mlir build + Add e2e spike |
| 3 | WS3 + WS4 | `ONNXSession` + op harness + ResNet spike |

---

## What not to change initially

- `pjrt_plugin_tt.so` compile API (`format == "mlir"`)
- `ModuleBuilder` SHLO → TTIR → TTNN pipeline
- PyTorch FX fusion / `tenstorrent.*` composites
- Multi-chip (defer until single-chip proof)
- Wheel packaging (onnx-mlir as dev tool first)

---

## Risk register

| Risk | Mitigation |
|------|------------|
| MLIR version mismatch (onnx-mlir vs tt-mlir) | Pin to tt-mlir LLVM commit; budget 1–3 weeks |
| tt-mlir op gaps on ONNX-lowered graphs | Op-by-op harness early; file gaps against tt-mlir |
| No composite ops from ONNX | Expected perf gap; Phase 5 SHLO fusion passes |
| Dynamic shapes | Specialize at compile time initially |
