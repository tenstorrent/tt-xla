---
name: debug-bfp4-accuracy
description: Debug accuracy issues in bfp4-quantized models by CPU-bypassing matmul operations to isolate quantization loss from TT kernel implementation issues. Use when bfp4 weights degrade accuracy and you need to determine if the cause is quantization or the hardware matmul kernel.
allowed-tools: Bash Read Grep Glob Write Edit
---

# Debug BFP4 Accuracy via CPU Matmul Bypass

Diagnoses whether accuracy degradation in bfp4-quantized models comes from:
1. Pure information loss of 4-bit quantization (inherent), OR
2. TT hardware matmul kernel implementation issues (accumulation precision, rounding, etc.)

## How It Works

Take a model with bfp4 weights, generate its device code via `codegen_py`, then modify the generated code to run bfp4 matmuls on CPU instead of TT hardware. The bfp4 quantization still happens on device (preserving information loss), but the matmul computation uses `torch.matmul` on CPU (eliminating kernel issues). PCC comparison between the two reveals whether the kernel or quantization is the bottleneck.

## Prerequisites

- Working tt-xla build (`source venv/activate` must succeed)
- TT hardware available (codegen_py requires device compilation)
- The target model must compile successfully with tt-xla

## Phase 1: Gather Information

Ask the user for:
1. **Which model?** (e.g., GPT-OSS-120B, Llama-3.1-8B)
2. **Which test function?** (e.g., `test_gpt_oss_120b` in `tests/benchmark/test_llms.py`)
3. **What weight_dtype_overrides?** (e.g., `{"model.layers.*.mlp.experts.gate_up_proj": "bfp_bf4", "model.layers.*.mlp.experts.down_proj": "bfp_bf4"}`)
4. **Single-chip or multi-chip?**
5. **How many layers?** (use fewer for faster iteration, e.g., `--num-layers 1`)

Read the test function in `tests/benchmark/test_llms.py` to get the exact configuration, model loader import, and variant.

## Phase 2: Write the Test Script

Write a standalone Python script that uses `run_codegen_cpu_bypass()` from `tests/benchmark/benchmarks/codegen_accuracy.py`. This module provides the full pipeline:

1. `codegen_py()` generates device code with real weights
2. `find_bfp4_matmul_groups()` parses the generated code for bfp4 matmul patterns
3. `generate_cpu_bypass_code()` creates a modified version with CPU matmul replacements
4. `run_and_compare()` executes both and computes PCC

### Script Template

```python
"""Debug bfp4 accuracy for [MODEL_NAME]."""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "tests/benchmark"))

import torch
import torch_xla
import torch_xla.runtime as xr
from tt_torch.weight_dtype import apply_weight_dtype_overrides

xr.set_device_type("TT")

# 1. Load model (copy from the test function)
from third_party.tt_forge_models.MODEL.pytorch.loader import ModelLoader, ModelVariant

model_loader = ModelLoader(num_layers=NUM_LAYERS)  # Use 1 for quick test
model = model_loader.load_model(dtype_override=torch.bfloat16)
model.eval()

# 2. Apply weight dtype overrides
weight_dtype_overrides = {
    # Copy from the test function
}
applied = apply_weight_dtype_overrides(model, weight_dtype_overrides)
print(f"Applied {len(applied)} weight dtype overrides")

# 3. Construct sample inputs
tokenizer = model_loader.tokenizer
inputs = tokenizer("Hello world", return_tensors="pt", max_length=128, truncation=True)
input_ids = inputs["input_ids"]

# 4. Run the codegen CPU bypass pipeline
from benchmarks.codegen_accuracy import run_codegen_cpu_bypass

results = run_codegen_cpu_bypass(
    model=model,
    sample_inputs=(input_ids,),
    sample_kwargs={},
    export_path="bfp4_debug_output",
    compiler_options={
        "optimization_level": 1,
    },
)

print(f"\nFinal PCC: {results['overall_pcc']:.6f}")
print(f"Interpretation: {results['interpretation']}")
```

Adapt this template to match the specific model's loader, inputs, and compiler options.

**CRITICAL**: `codegen_py` is called with `export_tensors=True` (the default) which serializes actual HuggingFace model weights to disk. This is essential -- random/ones weights make the accuracy analysis meaningless.

### For LLM Models with StaticCache

LLMs need `past_key_values` (StaticCache) and `cache_position` as inputs. The `codegen_py` function filters kwargs to only move `torch.Tensor` values. Wrap the model to internalize the cache:

```python
from llm_utils import init_static_cache

class ModelWithCache(torch.nn.Module):
    def __init__(self, model, config, batch_size, max_cache_len):
        super().__init__()
        self.model = model
        self.cache = init_static_cache(
            config=config, batch_size=batch_size,
            max_cache_len=max_cache_len, device="cpu", dtype=torch.bfloat16
        )

    def forward(self, input_ids, cache_position):
        return self.model(
            input_ids=input_ids,
            past_key_values=self.cache,
            cache_position=cache_position,
            use_cache=True,
        )

wrapped = ModelWithCache(model, model.config, batch_size=1, max_cache_len=128)
cache_position = torch.arange(0, input_ids.shape[1])

results = run_codegen_cpu_bypass(
    model=wrapped,
    sample_inputs=(input_ids, cache_position),
    sample_kwargs={},
    export_path="bfp4_debug_output",
)
```

## Phase 3: Run and Analyze

Run the script:
```bash
source venv/activate
python bfp4_debug_script.py 2>&1 | tee bfp4_debug.log
```

The output will include:
- Number of bfp4 matmul groups found
- Per-output PCC between device and CPU-bypassed runs
- Overall PCC
- Interpretation

## Phase 4: Interpret Results

### Generated Code Patterns

The generated `main.py` uses these exact patterns for bfp4 matmuls:
```python
var_X = ttnn.typecast(var_W, ttnn.DataType.BFLOAT4_B, ...)   # weight quantized
var_Y = ttnn.matmul(var_A, var_X, ...)                        # matmul with bfp4
var_Z = ttnn.typecast(var_Y, ttnn.DataType.BFLOAT16, ...)     # cast result back
```

The CPU bypass keeps the typecast-to-bfp4 (preserving quantization loss) but replaces the matmul with:
```python
# Move to CPU, cast bfp4 weight to bf16, do torch.matmul, move result back
```

### Result Interpretation

| PCC Range | Meaning | Action |
|-----------|---------|--------|
| >= 0.995 | Kernel and CPU produce nearly identical results | Accuracy loss is from quantization itself. Try keeping sensitive layers at bfp8 or bf16. |
| 0.99 - 0.995 | Moderate kernel contribution | Both factors matter. Try `fp32_dest_acc_en=True` or higher math fidelity. |
| < 0.99 | Kernel adds significant error | The TT matmul kernel is the primary issue. Investigate accumulation precision, math fidelity, tile padding. |

### If No BFP4 Matmul Groups Found

If the parser finds 0 groups, the weight_dtype_overrides may not have been applied before codegen, or the compiler optimized away the typecasts. Check:
1. That `apply_weight_dtype_overrides()` was called before `codegen_py()`
2. That the override patterns match actual parameter names
3. Inspect the generated `main.py` manually for `BFLOAT4_B` occurrences

## Key Files

| File | Purpose |
|------|---------|
| `tests/benchmark/benchmarks/codegen_accuracy.py` | Core module: `run_codegen_cpu_bypass()`, `find_bfp4_matmul_groups()`, `generate_cpu_bypass_code()`, `run_and_compare()` |
| `python_package/tt_torch/codegen.py` | `codegen_py()` function that generates ttnn Python code |
| `python_package/tt_torch/weight_dtype.py` | `apply_weight_dtype_overrides()` for per-tensor bfp4/bfp8 overrides |
| `tests/benchmark/benchmarks/llm_benchmark.py` | Reference for model setup: `setup_model_and_tokenizer()`, `construct_inputs()` |
| `tests/benchmark/utils.py` | `compute_pcc()`, `create_model_loader()` |
| `third_party/tt-mlir/.../templates/python/local/ttir_cpu.py` | CPU implementations of TTIR ops (reference) |

## Common Pitfalls

1. **Must use real weights**: `export_tensors=True` in `codegen_py` is the default and serializes actual model weights. If you see `ttnn.ones` in generated code, something went wrong.
2. **Keep the on-device bfp4 typecast**: The CPU bypass module preserves the `ttnn.typecast(..., ttnn.DataType.BFLOAT4_B)` call. Removing it would defeat the purpose.
3. **ttnn.to_torch on bfp4 tensors**: May fail. The bypass code typecasts to BFLOAT16 on device before calling `ttnn.to_torch`.
4. **Memory**: Moving all bfp4 weights to CPU may cause OOM on large models. Consider testing with `--num-layers 1` first.
5. **Multi-chip**: If tensors are sharded, add `ttnn.all_gather` before `ttnn.from_device`. Check if `create_inputs_for_forward` uses mesh_shape != (1,1).
6. **Model must be on CPU**: Call `codegen_py` before `model.to(device)`.

## Example: GPT-OSS-120B (1 Layer)

```python
from third_party.tt_forge_models.gpt_oss.pytorch.loader import ModelLoader, ModelVariant

model_loader = ModelLoader(num_layers=1)
model = model_loader.load_model(dtype_override=torch.bfloat16)
model.eval()

apply_weight_dtype_overrides(model, {
    "model.layers.*.mlp.router.weight": "bf16",
    "model.layers.*.mlp.experts.gate_up_proj": "bfp_bf4",
    "model.layers.*.mlp.experts.down_proj": "bfp_bf4",
})

# Wrap for LLM (see Phase 2 for full wrapper)
# ...

results = run_codegen_cpu_bypass(model=wrapped, ...)
```
