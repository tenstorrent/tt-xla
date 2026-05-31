---
name: ONNX Workstreams
overview: Implementation plan for native ONNX support on tt-xla via onnx-mlir bridge and PJRT integration.
parent_plan: onnx_support_options.md
isProject: false
---

# ONNX on TT-XLA — Workstreams & Implementation Plan

**Parent plans:** [onnx_support_options.md](onnx_support_options.md), [onnx_integration_milestones.md](onnx_integration_milestones.md), [onnx_implementation_log.md](onnx_implementation_log.md) (procedure, blockers, status)

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
**Status:** ✅ Complete (`a52b27974`)  
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

- [x] Unit tests pass for StableHLO and VHLO detection
- [x] Existing JAX/PyTorch compile path unchanged (VHLO still deserializes)
- [x] Hand-fed StableHLO MLIR reaches `runFrontendSHLOPipeline` without error
- [x] Documented compile option available for debugging

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
**Status:** ✅ Complete — smoke test passed (bundled LLVM; tt-mlir MLIR pin incompatible)

### Goal

Reproducible `onnx-mlir-opt` binary for ONNX → StableHLO lowering.

### Important: MLIR pin strategy

Building onnx-mlir against `/opt/ttmlir-toolchain` MLIR **fails** for commit `4400cbc` with:

- `Krnl.td`: mlir-tblgen `Unexpected overlap` (newer tblgen vs old Krnl dialect)
- Bundled `stablehlo` submodule: MLIR API mismatches (`QuantTypes.h`, `Type::dyn_cast`, float8 helpers)

**Fix:** `build_onnx_mlir.sh` builds the **LLVM revision pinned in onnx-mlir's `utils/clone-mlir.sh`**, then builds onnx-mlir against that MLIR. Output StableHLO is fed to tt-xla via WS1 (`mlir_input_format=auto`).

### Tooling (added)

| Path | Purpose |
|------|---------|
| `tools/onnx/build_onnx_mlir.sh` | Clone/build onnx-mlir against bundled LLVM (see MLIR pin strategy) |
| `tools/onnx/smoke_test.sh` | Add ONNX → StableHLO smoke test |
| `tools/onnx/gen_add_onnx.py` | Generate trivial Add ONNX fixture |
| `tools/onnx/onnx_mlir_commit.txt` | Pinned onnx-mlir git commit |
| `tools/onnx/README.md` | Full instructions |

### Steps

1. Read MLIR commit from `third_party/tt-mlir` submodule
2. Add `third_party/onnx-mlir` or `tools/onnx/build_onnx_mlir.sh`
3. Smoke test: 1-op Add ONNX → StableHLO MLIR
4. Document env var `TT_ONNX_MLIR_OPT` pointing to built binary

### Run (on reservation host with working venv)

```bash
cd /path/to/tt-xla
source venv/activate
pip install onnx numpy   # fixture generation only

tools/onnx/build_onnx_mlir.sh
source tools/onnx/env.sh
tools/onnx/smoke_test.sh
```

### Exit criteria

- [x] `onnx-mlir-opt --convert-onnx-to-stablehlo` runs on trivial Add model
- [x] Output parses as MLIR with `stablehlo` ops

See [onnx_implementation_log.md](onnx_implementation_log.md) for blockers, fixes, and runbook.

**Effort:** 1–3 weeks (MLIR pin alignment is the long pole)

---

## Workstream 3 — `tt_onnx` Python package

**Milestone:** M1.5  
**Prerequisite:** Workstreams 1 + 2 ✅  
**Status:** ✅ Complete — Add e2e passed on reservation host (TT device vs ORT CPU, `max_abs_diff=0`)

### Goal

Single Python API and spike script: ONNX → onnx-mlir → canonicalized StableHLO → PJRT compile → TT device execute, with ONNX Runtime CPU reference checks.

### Problem

onnx-mlir StableHLO is not directly consumable by the JAX tt compile path:

1. Emits `shape`/`arith`/`onnx.EntryPoint` ops that jaxlib’s MLIR context does not register.
2. Entry function is `@main_graph` (private by default); XLA expects `@main` and tt-xla needs one **public** `func.func`.
3. JAX `backend_compile_and_load` serializes input to **VHLO** before PJRT — forcing `mlir_input_format=stablehlo` skips VHLO→StableHLO and breaks `getPublicFuncOps`.

### Solution

1. **`mlir_utils.py`** — canonicalize onnx-mlir output before compile:
   - Strip `onnx.EntryPoint`
   - For static-shape Add: collapse shape-broadcast prelude to pure `stablehlo.add`
   - Rename entry `@main_graph` → `@main`; mark `func.func public`
2. **`compiler.py`** — feed canonical MLIR **text** to `backend.compile_and_load()` (not `ir.Module` + `module_to_bytecode`, which drops visibility)
3. **Compile option** `mlir_input_format=auto` — JAX sends VHLO; tt-xla auto-detects and runs VHLO deserialize (WS1)
4. **`ONNXSession`** — bridge (WS2 CLI) + compile + execute; compare vs ORT CPU in spike/test

### Tooling (added)

| Path | Purpose |
|------|---------|
| `python_package/tt_onnx/bridge.py` | ONNX → StableHLO via `onnx-mlir` / `onnx-mlir-opt` (same as `smoke_test.sh`) |
| `python_package/tt_onnx/mlir_utils.py` | Canonicalize onnx-mlir StableHLO for JAX/tt-xla |
| `python_package/tt_onnx/compiler.py` | StableHLO MLIR → PJRT `LoadedExecutable` via JAX tt backend |
| `python_package/tt_onnx/runtime.py` | Execute loaded executable + ORT CPU reference |
| `python_package/tt_onnx/session.py` | `ONNXSession` high-level API |
| `python_package/tt_onnx/plugin_check.py` | Preflight `dlopen` of `pjrt_plugin_tt.so` |
| `tools/onnx_spike/compile_add.py` | Add e2e spike script |
| `tests/onnx/test_add_e2e.py` | Pytest: TT device vs ORT CPU |

### Steps

1. Ensure WS1 + WS2 complete (`tools/onnx/smoke_test.sh` passes).
2. Rebuild PJRT if tt-mlir changed: `cmake --build build --target TTPJRTTTDylib -j$(nproc)`.
3. Confirm `python_package/pjrt_plugin_tt/pjrt_plugin_tt.so` symlink points at `build/pjrt_implementation/src/pjrt_plugin_tt.so`.
4. Install Python deps: `pip install onnx numpy onnxruntime`.
5. Run spike or pytest on reservation host (venv + `tools/onnx/env.sh`).

### Run (on reservation host with working venv)

```bash
cd /path/to/tt-xla
source venv/activate
source tools/onnx/env.sh
pip install onnx numpy onnxruntime

python tools/onnx_spike/compile_add.py 2>&1 | tee onnx_compile_add.log
pytest -svv tests/onnx/test_add_e2e.py
```

**Expected spike output:** `PASSED: Add ONNX e2e on TT device.` with `max_abs_diff=0` vs ORT CPU.

### API

```python
from tt_onnx import ONNXSession

session = ONNXSession("model.onnx", device="tt")
outputs = session.run({"A": arr_a, "B": arr_b})
```

Compile options passed to PJRT (via `ONNXSession`):

```python
{
    "export_path": "tools/onnx/build/tt_onnx/add_spike/export",
    "export_model_name": "add",
    "mlir_input_format": "auto",  # required when compiling via JAX (VHLO round-trip)
    "optimization_level": "0",
}
```

**Canonical Add MLIR shape** (after `mlir_utils`):

```mlir
module attributes {sym_name = "onnx_module"} {
  func.func public @main(%arg0: tensor<1x4xf32>, %arg1: tensor<1x4xf32>) -> tensor<1x4xf32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<1x4xf32>
    return %0 : tensor<1x4xf32>
  }
}
```

### Exit criteria

- [x] Single command: Add ONNX → compile → execute on TT device
- [x] Numerical match vs ONNX Runtime CPU reference
- [x] IR dumps under `export_path` (`vhlo_*`, `shlo_*`, `shlo_compiler_*`)

See [onnx_implementation_log.md](onnx_implementation_log.md) for blockers, fixes, and runbook.

**Effort:** 1–2 weeks (actual: ~1 week including compile-path debugging)

---

## Workstream 4 — Tests and incremental validation

**Milestones:** M1.6–M1.11  
**Prerequisite:** Workstream 3 ✅  
**Status:** 🔄 In progress — MLP e2e + seed op matrix harness (onnx-mlir only; torch-mlir comparison deferred)

### Scope (this phase)

- **Bridge:** onnx-mlir only (WS2/WS3 path). No torch-mlir dual-bridge runs.
- **M1.6 (partial):** 2-layer **elementwise** MLP e2e (`Mul→Add→Relu→Mul→Add`; no Gemm). Gemm MLP blocked on `stablehlo.dot` → TTIR (see implementation log).
- **M1.7 (seed):** Op matrix harness + 5 ops: Add, Mul, MatMul, Relu, Reshape. ✅
- **M1.8 (in progress):** Full 17-op matrix — Sub, Div, Sigmoid, ReduceMean, ReduceSum, Transpose, Concat, Slice, Conv, LayerNorm, Softmax, Gather.
- **Next:** Triage M1.8 failures, then ResNet-18 (M1.9).

### Tooling (added)

| Path | Purpose |
|------|---------|
| `tools/onnx/gen_mlp_onnx.py` | Generate 2-layer MLP fixture (`Gemm` → `Relu` → `Gemm`) |
| `tools/onnx/gen_op_onnx.py` | Generate single-op ONNX fixtures (seed + M1.8 full matrix) |
| `python_package/tt_onnx/op_matrix.py` | Op matrix runner + JSON report |
| `tools/onnx_spike/compile_mlp.py` | MLP e2e spike script |
| `tools/onnx_spike/run_op_matrix.py` | Seed/full op matrix CLI + JSON report |
| `tests/onnx/onnx_e2e_utils.py` | Shared e2e helpers |
| `tests/onnx/test_mlp_e2e.py` | Pytest: MLP on TT vs ORT |
| `tests/onnx/test_op_matrix_seed.py` | Pytest: 5 seed ops parametrized |
| `tests/onnx/test_op_matrix_full.py` | Pytest: M1.8 full 17-op matrix |

### Steps

1. Complete WS3 Add e2e on reservation host.
2. Generate fixtures: `gen_mlp_onnx.py`, `gen_op_onnx.py --all`.
3. Run MLP spike, then seed op matrix (records JSON report).
4. Run pytest push tests on reservation host.
5. Triage failures using `export/irs/shlo_compiler_*.mlir` + `tests/op_by_op/` if needed.

### Run (on reservation host)

```bash
cd /path/to/tt-xla
source venv/activate
source tools/onnx/env.sh
pip install onnx numpy onnxruntime

# Generate fixtures
python tools/onnx/gen_mlp_onnx.py
python tools/onnx/gen_op_onnx.py --all

# Spikes
python tools/onnx_spike/compile_mlp.py 2>&1 | tee onnx_compile_mlp.log
python tools/onnx_spike/run_op_matrix.py --report tools/onnx/build/tt_onnx/op_matrix/report.json

# Pytest
pytest -svv tests/onnx/test_mlp_e2e.py
pytest -svv tests/onnx/test_op_matrix_seed.py
```

### Order

| Step | Test | Proves | Status |
|------|------|--------|--------|
| 1 | Add ONNX → SHLO dump | Bridge works | ✅ WS2 |
| 2 | Add ONNX → PJRT compile-only | SHLO ingestion + tt-mlir | ✅ WS3 |
| 3 | Add ONNX → device execute | Full stack | ✅ WS3 |
| 4 | 2-layer MLP ONNX | Multi-op graph | 🔄 WS4 |
| 5 | Op matrix (5 seed ops) | Scaling signal | ✅ WS4 |
| 6 | Op matrix (17 full ops, M1.8) | Gap inventory | 🔄 WS4 |
| 7 | ResNet-18 from `tt_forge_models` | First real model | ⬜ |

Reuse existing infra:

- IR dumps via `export_path` / `export_model_name` (same as `examples/pytorch/export_ir_example.py`)
- Op failures via `tests/op_by_op/` on dumped `shlo_compiler_*.mlir`
- Model loaders from `third_party/tt_forge_models/*/onnx/loader.py`

### Exit criteria (WS4 phase 1)

- [ ] MLP e2e passes on TT device vs ORT CPU
- [ ] Seed op matrix JSON report with pass/fail per op
- [ ] Pytest `test_mlp_e2e` + `test_op_matrix_seed` green on reservation host

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
