# ONNX bridge tooling (Workstream 2)

Build **onnx-mlir** using **onnx-mlir's bundled LLVM/MLIR pin**, then smoke-test ONNX → StableHLO lowering.

**Why not `TTMLIR_TOOLCHAIN_DIR`?** tt-mlir's MLIR is too new for onnx-mlir@4400cbc (Krnl tblgen + stablehlo API errors). The build script compiles the LLVM revision from `utils/clone-mlir.sh` instead (~30–90 min first time).

This is milestone **M1.3** from [onnx_integration_milestones.md](../../docs/plans/onnx_integration_milestones.md).

## Prerequisites

- tt-xla environment: `source venv/activate`
- `clang++-20` or `clang++`, `ninja`, `cmake` (from venv)
- Python packages for fixture gen: `pip install onnx numpy`
- **Time:** first run builds LLVM + onnx-mlir (plan for 1–2 hours)

## Quick start

```bash
cd /path/to/tt-xla
source venv/activate
pip install onnx numpy

tools/onnx/build_onnx_mlir.sh
source tools/onnx/env.sh
tools/onnx/smoke_test.sh
```

## Environment variables

| Variable | Purpose |
|----------|---------|
| `ONNX_MLIR_COMMIT` | Override pinned onnx-mlir git commit |
| `ONNX_MLIR_BUILD_JOBS` | Parallel build jobs (default: `nproc`) |
| `ONNX_MLIR_FORCE_LLVM_REBUILD=1` | Rebuild bundled LLVM from scratch |
| `ONNX_MLIR_USE_TTMLIR_MLIR=1` | Experimental tt-mlir MLIR path (usually fails) |
| `TT_ONNX_MLIR_OPT` | Path to `onnx-mlir-opt` (set by `env.sh`) |
| `TT_ONNX_MLIR` | Path to `onnx-mlir` frontend driver |

Pinned commit: [onnx_mlir_commit.txt](onnx_mlir_commit.txt)

## Pipeline

```text
add.onnx
  └─ onnx-mlir --EmitONNXIR -o add
       └─ add.onnx.mlir  (ONNX dialect; -o must not include .onnx.mlir)
            └─ onnx-mlir-opt --convert-onnx-to-stablehlo
                 └─ add.stablehlo.mlir  (stablehlo.* ops)
                      └─ tt-xla PJRT (WS1 direct SHLO ingestion)
```

## Troubleshooting

### `add.onnx.mlir`: No such file or directory (smoke test)

`onnx-mlir --EmitONNXIR` treats `-o` as a **basename without extension** and appends
`.onnx.mlir`. Passing `-o .../add.onnx.mlir` produces `add.onnx.mlir.onnx.mlir` instead.
`smoke_test.sh` uses `-o .../add`; if you hit this after an older script version, remove
stale artifacts and re-run:

```bash
rm -rf tools/onnx/build/smoke
tools/onnx/smoke_test.sh
```

### MLIR version mismatch (tt-mlir toolchain)

If compile fails with `Krnl.td: Unexpected overlap`, `QuantTypes.h: No such file`, or
`Type has no member named isFloat8E4M3FN`, you are building against tt-mlir's MLIR.
Use the default bundled-LLVM path (do **not** set `ONNX_MLIR_USE_TTMLIR_MLIR=1`):

```bash
rm -rf tools/onnx/build/onnx-mlir-build tools/onnx/build/onnx-mlir-src/llvm-project/build
tools/onnx/build_onnx_mlir.sh
```

First bundled-LLVM build takes 30–90+ minutes.

### `cmake --install` / `librapidcheck.a` missing

With `ONNX_MLIR_BUILD_TESTS=OFF`, full install fails because test deps like
`rapidcheck` are not built. The script copies only `onnx-mlir-opt` and
`onnx-mlir` from the build tree instead. If compile already finished, run:

```bash
BIN=tools/onnx/build/onnx-mlir-build/Release/bin
mkdir -p tools/onnx/build/install/bin
install -m 755 "$BIN/onnx-mlir-opt" "$BIN/onnx-mlir" tools/onnx/build/install/bin/
source tools/onnx/env.sh   # or re-run build_onnx_mlir.sh (skips LLVM if cached)
tools/onnx/smoke_test.sh
```

### Missing submodule / `ChloOps` / `pybind11_add_module` errors

If configure fails with missing `third_party/stablehlo`, `pybind11`, `benchmark`, or
`install TARGETS given target "ChloOps" which does not exist`, submodules were not
initialized. The build script now runs:

```bash
git submodule update --init third_party/onnx third_party/pybind11 \
  third_party/rapidcheck third_party/stablehlo third_party/benchmark
```

Clean and rebuild:

```bash
rm -rf tools/onnx/build/onnx-mlir-build tools/onnx/build/onnx-mlir-src
tools/onnx/build_onnx_mlir.sh
```

### CMake 4.x / `cmake_minimum_required` error

If configure fails in `third_party/onnx/CMakeLists.txt` with "Compatibility with CMake < 3.5 has removed", the build script passes `-DCMAKE_POLICY_VERSION_MINIMUM=3.5` automatically. Re-run:

```bash
rm -rf tools/onnx/build/onnx-mlir-build
tools/onnx/build_onnx_mlir.sh
```

### MLIR version mismatch

onnx-mlir must compile against **the same MLIR** as tt-mlir. Do not build a separate llvm-project unless MLIR ABI errors appear. Try a different `ONNX_MLIR_COMMIT` from [onnx/onnx-mlir](https://github.com/onnx/onnx-mlir).

### `llvm-lit` not found

Ensure `$TTMLIR_TOOLCHAIN_DIR/bin/llvm-lit` exists, or set `-DLLVM_EXTERNAL_LIT=` to your lit path when re-running cmake manually.

### Build artifacts

Local build outputs live under `tools/onnx/build/` (gitignored). Safe to delete and rebuild:

```bash
rm -rf tools/onnx/build
tools/onnx/build_onnx_mlir.sh
```

## Exit criteria (WS2)

- [x] `build_onnx_mlir.sh` completes
- [x] `smoke_test.sh` passes (`stablehlo.add` in output)
- [x] `TT_ONNX_MLIR_OPT` documented and usable for WS3 (`tt_onnx` package)

## Workstream 4 — MLP + op matrix (onnx-mlir only)

After WS3 Add e2e passes, generate fixtures and run validation on the reservation host:

```bash
source venv/activate
source tools/onnx/env.sh
pip install onnx numpy onnxruntime

python tools/onnx/gen_mlp_onnx.py
python tools/onnx/gen_op_onnx.py --all

python tools/onnx_spike/compile_mlp.py
python tools/onnx_spike/run_op_matrix.py --report tools/onnx/build/tt_onnx/op_matrix/report.json

pytest -svv tests/onnx/test_mlp_e2e.py tests/onnx/test_op_matrix_seed.py
```

Seed ops: `add`, `mul`, `matmul`, `relu`, `reshape`.

**M1.8 full matrix (17 ops):**

```bash
python tools/onnx/gen_op_onnx.py --all --full
python tools/onnx_spike/run_op_matrix.py --full
pytest -svv tests/onnx/test_op_matrix_full.py -m push

# One-op triage (bridge → canonical MLIR → compile → e2e)
python tools/onnx_spike/triage_op.py reduce_mean
python tools/onnx_spike/triage_op.py gather --bridge-only
```

Full ops add: `sub`, `div`, `sigmoid`, `reduce_mean`, `reduce_sum`, `transpose`, `concat`, `slice`, `conv`, `layer_norm`, `softmax`, `gather`. Report: `tools/onnx/build/tt_onnx/op_matrix_full/report.json`.

See [onnx_workstreams.md](../../docs/plans/onnx_workstreams.md) WS4.
