---
name: debug-pcc
description: >
  Debug PCC (Pearson Correlation Coefficient) failures in tt-xla performance benchmarks.
  Use this skill whenever a benchmark test reports a PCC drop or assertion failure,
  when the user mentions PCC issues, numerical divergence, precision problems with
  optimized models, or wants to find which TTNN op breaks accuracy. Also trigger when
  the user mentions instrument_codegen, vanillify, or codegen PCC comparison.
  Covers the full workflow: generating codegen Python, running the instrumentation
  tool to compare vanilla vs optimized intermediate tensors, identifying the first
  bad op, and creating an isolated reproduction test.
---

# Debug PCC Issues in TT-XLA Benchmarks

This skill walks through debugging PCC (Pearson Correlation Coefficient) failures
in tt-xla benchmark tests. PCC measures how closely device outputs match CPU
golden outputs. When the tt-mlir compiler applies optimizations (level 1 or 2),
it may move tensors from DRAM to L1 memory, change memory layouts, or fuse
operations -- any of which can introduce numerical divergence.

## When to use this

- A benchmark test fails its PCC assertion (e.g. expected >= 0.97, got 0.91)
- PCC regressed after a tt-mlir or tt-metal update
- A new model has low PCC only at optimization_level > 0
- You need to isolate which specific TTNN operation causes precision loss

## High-level workflow

1. **Reproduce the PCC failure** with the normal benchmark
2. **Generate codegen Python** by switching the benchmark backend to `codegen_py`
3. **Run `instrument_codegen.py auto`** to compare vanilla vs optimized per-op
4. **Identify the first bad op** from the comparison summary
5. **Create an isolated reproduction** for the offending op

---

## Step 1: Reproduce the PCC failure

Run the failing benchmark to confirm the issue and capture the PCC value:

```bash
source venv/activate
pytest -svv tests/benchmark/test_vision.py::test_swin
```

Look for output like:
```
AssertionError: PCC 0.9134 is below required threshold 0.97
```

Note the model name, optimization level, and reported PCC.

The benchmark test files live in `tests/benchmark/`:
- `test_vision.py` -- vision models (resnet50, swin, efficientnet, unet, etc.)
- `test_llms.py` -- language models (llama, gemma, qwen, ministral, etc.)
- `test_encoders.py` -- encoder models

Each test calls a benchmark function in `tests/benchmark/benchmarks/` which sets
compile options via `torch_xla.set_custom_compile_options(options)`.

### Speeding things up with fewer layers

For LLM models, you can reduce the number of transformer layers to speed up both
reproduction and codegen. The benchmark harness supports `--num-layers`:

```bash
pytest -svv tests/benchmark/test_llms.py::test_ministral_8b --num-layers 2
```

This compiles only 2 layers instead of the full model, which is much faster and
often sufficient to reproduce the PCC issue since the same optimization decisions
apply to every layer. Start small (1-2 layers) and increase only if the PCC
failure doesn't reproduce.

Vision models (test_vision.py) do not currently support `--num-layers` -- they
always compile the full model. For large vision models, the codegen step may take
longer but there's no shortcut available.

## Step 2: Generate codegen Python

To get a standalone TTNN Python script we can instrument, modify the benchmark's
compile options to emit Python code instead of a flatbuffer binary.

Find the options dict in the relevant benchmark file. For vision models it's in
`tests/benchmark/benchmarks/vision_benchmark.py` around line 151:

```python
options = {
    "optimization_level": optimization_level,
    "export_path": MODULE_EXPORT_PATH,
    "export_model_name": export_model_name,
    ...
}
```

Add these two keys to switch to codegen mode:

```python
options = {
    "optimization_level": optimization_level,
    "export_path": MODULE_EXPORT_PATH,
    "export_model_name": export_model_name,
    ...
    "backend": "codegen_py",        # emit Python instead of flatbuffer
    "export_tensors": True,          # serialize input tensors as .tensorbin
}
```

For LLM benchmarks, the options dict is in `tests/benchmark/benchmarks/llm_benchmark.py`.

Then run the benchmark again (with `--num-layers` for LLMs if desired). It will
compile the model and write generated files instead of executing on device:

```bash
# Vision
pytest -svv tests/benchmark/test_vision.py::test_swin

# LLM (use fewer layers for faster iteration)
pytest -svv tests/benchmark/test_llms.py::test_ministral_8b --num-layers 2
```

After this completes, the generated code appears under `modules/`:
```
modules/
  main.py              -- generated TTNN Python code (the optimized version)
  utils.py             -- codegen runtime utilities (DeviceGetter, tensor loading)
  tensors/             -- serialized input/weight tensors as .tensorbin files
  irs/                 -- intermediate representations (SHLO, TTIR, TTNN MLIR)
```

The `main.py` file contains direct `ttnn.*` API calls that mirror exactly what
the compiler would execute. The optimization level affects memory configs
(L1 vs DRAM), matmul program configs, and operation fusion in this generated code.

**Important:** After generating, revert the backend change so the benchmark runs
normally again. The codegen step only needs to happen once per investigation.

## Step 3: Run instrument_codegen.py

The `instrument_codegen.py` script (at `scripts/instrument_codegen.py`) compares
intermediate tensor outputs between a "vanilla" baseline and the optimized version
to pinpoint which operation introduces numerical divergence.

### What "vanillify" means

The script creates a vanilla baseline by:
- Converting ALL `ttnn.MemoryConfig(...)` to DRAM interleaved (removing L1/sharded configs)
- Replacing matmul `program_config` with `None` (letting TTNN auto-select kernels)
- Extracting `fused_activation` from program configs and moving it to the `activation=` kwarg
- Preserving sharded configs only where hardware requires it (e.g. `paged_update_cache` inputs)

This vanilla version has the same graph structure but uses the safest memory
configuration. If the vanilla version produces correct results but the optimized
version doesn't, the divergence is caused by the optimizer's memory/layout decisions.

### The `auto` command (recommended)

```bash
python scripts/instrument_codegen.py auto modules/main.py
```

This runs three steps automatically:
1. Creates `modules/main_vanilla.py` (DRAM interleaved baseline)
2. Runs vanilla version, monkey-patches every TTNN compute op to save intermediate
   tensors as `.tensorbin` files in `golden_tensors/`
3. Runs the optimized `main.py` with the same monkey-patching, but instead of
   saving, compares each op's output against the golden tensor via PCC

The monkey-patching intercepts ~35 TTNN ops (matmul, linear, add, softmax,
rms_norm, to_memory_config, etc.) plus namespaced ops like
`ttnn.transformer.scaled_dot_product_attention`.

### Reading the output

The compare phase prints per-op PCC scores:

```
  [matmul_0]            PCC=0.999998 OK
  [add_0]               PCC=0.999999 OK
  [to_memory_config_3]  PCC=0.912345 BAD <<<
  [matmul_1]            PCC=0.891234 BAD <<<
```

The summary at the end shows:
```
SUMMARY
======================================================================
Total ops compared: 147
  OK   (PCC >= 0.99): 89
  WARN (0.95-0.99):   12
  BAD  (PCC < 0.95):  46

First bad op: to_memory_config_3
^ This is likely where the optimizer introduces the error.
```

The **first bad op** is the key finding. Everything downstream of it will also
show degraded PCC because the error cascades through the computation graph.

### Step-by-step alternative

If `auto` has issues (e.g. device reset between runs fails), run each step
separately:

```bash
# 1. Create vanilla version
python scripts/instrument_codegen.py vanillify modules/main.py

# 2. Run vanilla and dump golden tensors
python scripts/instrument_codegen.py dump modules/main_vanilla.py --tensor-dir golden_tensors

# 3. Compare optimized against golden
python scripts/instrument_codegen.py compare modules/main.py --tensor-dir golden_tensors
```

### Prerequisites

- `TT_MLIR_HOME` environment variable must be set (set by `source venv/activate`)
- The tt-xla virtual environment must be activated
- Generated `modules/main.py` and `modules/tensors/` must exist from Step 2

## Step 4: Identify the offending op

Once you have the first bad op (e.g. `to_memory_config_3`), open `modules/main.py`
and find that operation. The generated code is sequential, so you can count
occurrences of the op to find the exact line.

Common culprits:
- **`to_memory_config`** -- resharding from DRAM to L1 or between shard specs
  can introduce precision loss depending on the padding/alignment
- **`matmul` with specific `program_config`** -- certain matmul kernel configs
  at L1 can lose precision vs DRAM-based auto-selected kernels
- **`linear`** -- same as matmul (linear is matmul + bias under the hood)
- **`softmax`/`rms_norm`/`layer_norm`** -- reduction ops can be sensitive to
  memory layout changes

Look at the memory_config and program_config arguments of the failing op in
`main.py` vs what the vanilla version uses. The difference reveals what the
optimizer changed that broke precision.

## Step 5: Create an isolated reproduction

Extract the failing op and its inputs into a standalone test script. The goal is
a minimal reproducer that a tt-mlir or tt-metal developer can run independently.

Example structure for a matmul PCC issue:

```python
import ttnn
import torch

device = ttnn.open_device(0)

# Load or create input tensors matching the shapes from main.py
a = ttnn.from_torch(torch.randn(1, 1, 256, 512, dtype=torch.bfloat16),
                     layout=ttnn.TILE_LAYOUT, device=device)
b = ttnn.from_torch(torch.randn(1, 1, 512, 1024, dtype=torch.bfloat16),
                     layout=ttnn.TILE_LAYOUT, device=device)

# Run with DRAM interleaved (golden)
dram_config = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM)
result_golden = ttnn.matmul(a, b, memory_config=dram_config)

# Run with the optimizer's config (the one that breaks PCC)
l1_config = ttnn.MemoryConfig(
    ttnn.TensorMemoryLayout.WIDTH_SHARDED,
    ttnn.BufferType.L1,
    ttnn.ShardSpec(...)  # copy from main.py
)
result_opt = ttnn.matmul(a, b, memory_config=l1_config,
                          program_config=...)  # copy from main.py

# Compare
golden_torch = ttnn.to_torch(ttnn.from_device(result_golden))
opt_torch = ttnn.to_torch(ttnn.from_device(result_opt))
pcc = torch.corrcoef(torch.stack([golden_torch.flatten(), opt_torch.flatten()]))[0, 1]
print(f"PCC: {pcc:.6f}")

ttnn.close_device(device)
```

For real tensors, you can load the `.tensorbin` files from `modules/tensors/`
or `golden_tensors/` using `ttnn.load_tensor()`.

## PCC thresholds reference

| Threshold | Classification | Meaning |
|-----------|---------------|---------|
| >= 0.99   | OK            | Numerically equivalent within expected floating-point tolerance |
| 0.95-0.99 | WARN          | Minor divergence, may or may not be a problem depending on model |
| < 0.95    | BAD           | Significant divergence, likely a real optimization bug |

Default required PCC by model type (from benchmark tests):
- Vision models: 0.97
- LLMs: 0.95
- Encoders: 0.97
- Some models override lower (e.g. ResNet50, Swin at 0.90)

## Tips

- **Use `--num-layers` for LLMs** to drastically reduce iteration time. A 1-2
  layer model compiles and runs in seconds vs minutes for the full model, and
  the same optimizer decisions apply to every layer.
- If the first bad op is a `to_memory_config`, the issue is likely in the shard
  spec or memory layout chosen by the optimizer, not in the compute op itself.
- If multiple unrelated ops go bad simultaneously, the issue might be in a shared
  input tensor that got corrupted by an earlier resharding.
- For LLM models with tensor parallelism, make sure the codegen is run on the
  same chip configuration as the benchmark.
- The `modules/irs/` directory contains MLIR at various stages (SHLO, TTIR, TTNN).
  The TTNN MLIR shows the compiler's final decisions before code emission.
- You can edit `main.py` directly to test hypotheses -- e.g. change one op's
  memory_config back to DRAM interleaved and re-run to confirm it fixes PCC.
