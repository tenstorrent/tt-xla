# Mixed Precision (Per-Tensor Weight Dtype Overrides)

When uniform weight conversion (e.g. `experimental_weight_dtype: "bfp_bf8"`) causes accuracy degradation in specific layers, you can specify dtype overrides on a per-tensor basis. This lets you keep sensitive layers at higher precision (e.g. `bf16`) while converting the rest to a lower format (e.g. `bfp_bf8` or `bfp_bf4`).

> **Note:** Currently only matmul/linear layer weight overrides are propagated and respected. Convolution weights on lower data types are not yet supported through the compiler.

## Method 1: Python dict (recommended)

Build a dict mapping parameter names (or glob patterns) to target dtypes and call `apply_weight_dtype_overrides()`:

```python
from tt_torch import apply_weight_dtype_overrides

# Override specific weights by name.
apply_weight_dtype_overrides(model, {
    "fc2.weight": "bfp_bf8",
})

# Or use glob patterns to target groups of layers.
apply_weight_dtype_overrides(model, {
    "model.layers.*.mlp.gate_proj.weight": "bfp_bf4",
    "model.layers.*.mlp.up_proj.weight": "bfp_bf4",
    "model.layers.0.self_attn.q_proj.weight": "bf16",
})

# A "default" key applies to all weights, with specific overrides taking precedence.
apply_weight_dtype_overrides(model, {
    "default": "bfp_bf8",
    "model.layers.0.self_attn.q_proj.weight": "bf16",
})
```

Call this **after** creating the model and **before** `torch.compile`. See `examples/pytorch/mnist_performant.py` for a complete working example that lowers the last linear layer weight to bfp_bf8.

## Method 2: JSON config + CLI

For large models with hundreds of weight parameters, use the `tt-gen-weight-template` CLI to generate a JSON template, then edit it:

```bash
tt-gen-weight-template --loader third_party/tt_forge_models/llama/causal_lm/pytorch/loader.py
```

**CLI options:**

| Option | Description |
|--------|-------------|
| `--loader` | **(Required)** Path to a model `loader.py` file |
| `--variant` | Variant enum name (e.g. `LLAMA_3_1_8B`). Defaults to the loader's `DEFAULT_VARIANT` |
| `--list-variants` | List available variants and exit |
| `--default-dtype` | Default dtype for all entries: `bfp_bf8` (default), `bfp_bf4`, or `bf16` |
| `--output-dir` | Override output directory (default: `mixed_precision_configs/` next to the loader) |
| `--auto-class` | transformers `Auto*` class to use (default: `AutoModelForCausalLM`) |

The output is a JSON file mapping every weight parameter to a dtype string. Edit it to fine-tune per-layer dtypes:

```json
{
    "model.layers.*.mlp.gate_proj.weight": "bfp_bf4",
    "model.layers.*.mlp.up_proj.weight": "bfp_bf4",
    "model.layers.0.self_attn.q_proj.weight": "bf16"
}
```

Then pass the JSON file path to `apply_weight_dtype_overrides()`:

```python
apply_weight_dtype_overrides(model, "path/to/config.json")
```

**Auto-discovery for tests and benchmarks:** JSON configs placed in `mixed_precision_configs/` next to the model's `loader.py` are automatically discovered by the model test runner (`tests/runner/test_models.py`) and LLM benchmarks (`tests/benchmark/benchmarks/llm_benchmark.py`).

## In-model annotation

If you control the model code, you can annotate weights directly in the forward pass using `torch.ops.tt.weight_dtype_override`:

```python
def forward(self, x):
    w = torch.ops.tt.weight_dtype_override(self.fc.weight, "bfp_bf8")
    return torch.matmul(x, w)
```

This is useful for custom models or when you need dtype overrides to interact with other operations (e.g. tensor-parallel sharding). In practice this is rarely needed — the dict and JSON methods above cover most use cases.

## How it works

Overrides are applied transparently via `torch.nn.utils.parametrize` — there is **no need** to edit model forward functions or manually insert custom ops (unless using in-model annotation). The `apply_weight_dtype_overrides()` function registers a parametrization on each matched weight that injects a `torch.ops.tt.weight_dtype_override` call. During compilation, a C++ frontend pass extracts these annotations and propagates them as per-argument attributes for the tt-mlir weight dtype conversion pass.

> **Note:** If `apply_weight_dtype_overrides()` is called multiple times on the same model (e.g. first with a dict, then with a JSON config), the first call has priority for any given weight — already-parametrized weights are not overridden by subsequent calls.
