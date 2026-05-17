---
name: ttnn-ir-edit
description: Edit a compiled model's TTNN MLIR IR, regenerate a flatbuffer with `ttmlir-translate`, and re-run a pytest test that picks up the edited flatbuffer (no recompile). Use when the user wants to "tweak TTNN IR", "edit compute kernel config", "patch matmul knobs", or test hypotheses about op attributes (`compute_config`, math fidelity, fp32_dest_acc_en, memory configs, dtype, etc.) without touching the tt-mlir compiler. Validates the change end-to-end via the test's existing PCC comparison.
allowed-tools: Bash Read Edit Write Glob Grep
---

# TTNN IR Edit & Flatbuffer Reload

This skill lets you change op attributes (e.g., `#ttnn.device_compute_kernel_config<...>`) on a compiled TTNN module, regenerate a flatbuffer, and verify the change via an existing pytest with PCC.

## Background

The tt-xla PJRT plugin supports two compile options the skill relies on:
- `export_path` — dumps stage IRs to `<export_path>/irs/<stage>_<model_name>_<ts>.mlir` and the generated flatbuffer to `<export_path>/fb_<model_name>_<ts>.ttnn`. Set via `CompilerConfig(export_path=..., export_model_name=...)`.
- `flatbuffer_load_path` — when set and the file exists, the plugin **skips** `ttnnToFlatbuffer` and loads the binary from disk. Logged as `Loading flatbuffer from override path: <path>`. Set via `CompilerConfig(flatbuffer_load_path=...)`.

The MLIR pipeline still runs even when loading (so sharding metadata is verified). The test's PCC computation is unchanged — it just compares TT-device output against the CPU reference.

For a worked example see `tests/torch/ops/test_matmul.py::test_matmul_mf_fp32_acc`: the test sets `export_path="modules"` and `export_model_name="matmul_mf_fp32_acc_<mf>_<bool>"`, and auto-picks up `flatbuffers/<model_name>.ttnn` if present.

## Prerequisites

- tt-xla built (`pjrt_plugin_tt.so` exists) and `source venv/activate` works.
- `ttmlir-translate` available. Path:
  `/localdev/$USER/repos/tt-xla/third_party/tt-mlir/src/tt-mlir/build/bin/ttmlir-translate`
  (or wherever the user's tt-xla build dir is — check first).
- A wormhole/blackhole device available (test fails with no device).

## Workflow

### Step 1 — Identify the test and the knob

Ask the user (or infer from the conversation):
- Which test? (e.g., `tests/torch/ops/test_matmul.py::test_matmul_mf_fp32_acc[True-hifi2]`)
- Which op + which attribute? (e.g., `ttnn.matmul`'s `compute_config`, specifically `math_fidelity` and `fp32_dest_acc_en`).

If the test doesn't already wire export_path/flatbuffer_load_path, you must add that wiring first (see Step 2). If it does, jump to Step 3.

### Step 2 — Wire the test (only needed once per test)

Modify the test function to:
1. Compute a deterministic `model_name` from the test parameters.
2. Set `CompilerConfig(..., export_path="modules", export_model_name=model_name)`.
3. If `flatbuffers/<model_name>.ttnn` exists, set `compiler_config.flatbuffer_load_path = str(...)`.

Reference snippet (already in `tests/torch/ops/test_matmul.py`):

```python
from pathlib import Path
TTNN_IR_EXPORT_DIR = Path(__file__).resolve().parents[3] / "modules"
TTNN_FB_LOAD_DIR = Path(__file__).resolve().parents[3] / "flatbuffers"

model_name = f"matmul_mf_fp32_acc_{math_fidelity}_{fp32_dest_acc_en}"
compiler_config = CompilerConfig(
    math_fidelity=math_fidelity,
    fp32_dest_acc_en=fp32_dest_acc_en,
    export_path=str(TTNN_IR_EXPORT_DIR),
    export_model_name=model_name,
)
fb_path = TTNN_FB_LOAD_DIR / f"{model_name}.ttnn"
if fb_path.exists():
    compiler_config.flatbuffer_load_path = str(fb_path)
```

### Step 3 — Capture the baseline TTNN IR

`mkdir -p flatbuffers` (idempotent — keeps any previously-edited FBs).

Run the test once **without** any FB present for that `model_name`. It will compile normally and dump:
- `modules/irs/ttnn_<model_name>_g0_<ts>.mlir` ← edit this one
- `modules/irs/ttnn_runtime_<model_name>_g0_<ts>.mlir`
- `modules/fb_<model_name>_g0_<ts>.ttnn` ← reference FB (passing run)

```bash
source venv/activate
pytest -svv '<test_id>'
```

Confirm the `ttnn_*.mlir` was created and inspect the op of interest — find the line like:
```
"ttnn.matmul"(...) <{compute_config = #ttnn.device_compute_kernel_config<math_fidelity = hifi2, fp32_dest_acc_en = true>, ...}>
```

### Step 4 — Edit the IR

Copy the baseline `ttnn_*.mlir` into `flatbuffers/<model_name>.mlir` (a stable name you control). Edit *only* the attribute you want to change. Common knobs on `#ttnn.device_compute_kernel_config`:
- `math_fidelity = lofi | hifi2 | hifi3 | hifi4`
- `fp32_dest_acc_en = true | false`
- `packer_l1_acc`, `dst_full_sync_en` if exposed (see `include/ttmlir/Dialect/TTNN/IR/TTNNOpsAttrs.td` in tt-mlir).

Other useful attributes to tweak per op:
- Memory config / buffer type on `ttnn.to_layout`, `ttnn.matmul` outputs.
- Tile sizes / shard specs in `#ttnn_layout`.

Do NOT change op signatures (input/output shapes/types) — the plugin's verifier checks that the FB's program inputs/outputs match the sharding metadata derived from the MLIR module. Same shape, same dtype = same signature.

### Step 5 — Regenerate the flatbuffer

```bash
TTMLIR_TRANSLATE=/path/to/tt-mlir/build/bin/ttmlir-translate
$TTMLIR_TRANSLATE --ttnn-to-flatbuffer \
  flatbuffers/<model_name>.mlir \
  -o flatbuffers/<model_name>.ttnn
```

If `ttmlir-translate` errors, it's almost always a syntactic issue with the edited IR (mismatched attribute syntax, dangling locs, etc.). Read the error, fix the file, re-run.

### Step 6 — Re-run the test, verify

```bash
source venv/activate
TTXLA_LOGGER_LEVEL=DEBUG pytest -svv '<test_id>' 2>&1 | grep -E "Loading flatbuffer|PASSED|FAILED"
```

You must see:
- `Loading flatbuffer from override path: .../flatbuffers/<model_name>.ttnn` (confirms the override fired, not a recompile).
- `PASSED` or `FAILED` from pytest — the test's PCC threshold is the verdict.

If the test fails the PCC check, that's a real signal that your IR edit changed numerics in a way the test cares about. Report the result and the path of the FB used.

### Step 7 — Iterate or clean up

To try a different edit: re-edit `flatbuffers/<model_name>.mlir`, re-run `ttmlir-translate`, re-run pytest. The plugin will reload the new FB.

To revert to a compile-from-scratch run: delete `flatbuffers/<model_name>.ttnn` (or the whole `flatbuffers/` dir).

## Troubleshooting

- **`Loading flatbuffer from override path` does NOT appear**: the test isn't setting `flatbuffer_load_path`, or the file path is wrong. Verify `flatbuffers/<model_name>.ttnn` exists at the path the test computes, and that `model_name` matches.
- **`flatbuffer_load_path was set ... but the file does not exist; compiling from MLIR instead`**: the file is missing. Check the spelling and the `<model_name>` formatting.
- **Verifier failure (`Created flatbuffer binary contains different number of inputs/outputs`)**: you changed the op signature in the IR. Revert to same input/output types.
- **Test passes despite extreme edit**: the op is too small for the knob to matter (e.g., bf16 matmul of `64x64x64` will hit acceptable PCC even with `lofi`). Pick a larger test or a tighter `required_pcc`.

## Files touched by the underlying implementation (for reference)

- `pjrt_implementation/inc/api/compile_options.h` — `flatbuffer_load_path` field
- `pjrt_implementation/src/api/compile_options.cc` — option parsing
- `pjrt_implementation/inc/api/module_builder/module_builder.h` — `createFlatbufferBinary` signature
- `pjrt_implementation/src/api/module_builder/module_builder.cc` — load-vs-generate branch and caller
- `tests/infra/testers/compiler_config.py` — Python `CompilerConfig.flatbuffer_load_path`
- `tests/torch/ops/test_matmul.py` — example test wiring
