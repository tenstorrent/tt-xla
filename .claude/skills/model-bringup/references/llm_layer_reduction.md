# LLM Layer Reduction for Fast Bringup

Running an LLM with all layers during initial bringup is slow and wasteful — compilation and runtime failures almost always reproduce with just 1-2 transformer layers. This reference covers the `num_layers` pattern used across tt-forge-models loaders.

## Why reduce layers?

- **Compilation time**: A 70B model with 80 layers compiles the same graph structure as 1 layer, but takes orders of magnitude longer
- **Runtime memory**: Fewer layers means less device memory pressure, isolating real OOM from bringup OOM
- **Iteration speed**: Fix-recompile-test cycles go from minutes to seconds
- **Same failure surface**: Frontend compilation, tt-mlir compilation, and runtime errors reproduce identically with 1 layer — the graph structure per layer is the same

## When to use full layers

- **PCC/accuracy validation** — precision accumulates across layers, so final accuracy checks must use the full model
- **Memory fitting** — after single-layer passes, gradually increase to find the memory ceiling
- **Multi-layer interactions** — rare cases where KV cache or cross-layer state causes issues (test with 2 layers)

## The `num_layers` pattern

### Standard implementation in loader.py

```python
from typing import Optional
from transformers import AutoConfig, AutoModelForCausalLM

class ModelLoader(ForgeModel):
    def __init__(self, variant=None, num_layers: Optional[int] = None):
        """
        Args:
            variant: Model variant to load.
            num_layers: Optional number of hidden layers. If None, uses model default.
        """
        super().__init__(variant)
        self.num_layers = num_layers

    def load_model(self, *, dtype_override=None, **kwargs):
        pretrained_model_name = self._variant_config.pretrained_model_name

        model_kwargs = {}
        if dtype_override is not None:
            model_kwargs["torch_dtype"] = dtype_override
        model_kwargs |= kwargs

        # Override num_hidden_layers in the model config
        if self.num_layers is not None:
            config = AutoConfig.from_pretrained(pretrained_model_name)
            config.num_hidden_layers = self.num_layers
            model_kwargs["config"] = config

        model = AutoModelForCausalLM.from_pretrained(
            pretrained_model_name, **model_kwargs
        )
        model.eval()
        return model
```

### Key details

- The parameter is always called `num_layers` in the loader `__init__`
- It maps to `config.num_hidden_layers` for HuggingFace transformers models
- When `None` (default), the model loads with its full layer count — no behavior change
- Works with `AutoConfig.from_pretrained()` to get the base config, then override

### Models that already support `num_layers`

As of the current codebase, these loaders accept `num_layers`:
- `qwen_2/causal_lm/pytorch/loader.py`
- `qwen_2_5/causal_lm/pytorch/loader.py`
- `mistral/pytorch/loader.py`
- `mamba/pytorch/loader.py`
- `nanogpt/pytorch/loader.py`
- `roberta/pytorch/loader.py`

### Models where you should add it

Any HuggingFace transformer-based LLM loader that doesn't yet have it. The pattern is the same — add `num_layers` to `__init__` and the `AutoConfig` override to `load_model`.

For non-HuggingFace models, the config field may differ:
- Most HF transformers: `config.num_hidden_layers`
- GPT-style custom models: `config.n_layer` or `config.n_layers`
- Check the model's config class to find the right field

## Using `num_layers` via the test infrastructure

Always use the existing test infra rather than standalone scripts. The **benchmark test infra** (`tests/benchmark/`) supports `--num-layers` as a pytest CLI option.

### Benchmark tests with `--num-layers`

The `--num-layers` option is defined in `tests/benchmark/conftest.py` and injected as a `num_layers` pytest fixture. Benchmark tests use `create_model_loader(ModelLoader, num_layers=num_layers)` from `tests/benchmark/utils.py`, which:
1. Checks if the `ModelLoader.__init__` accepts a `num_layers` parameter
2. If yes, passes it through
3. If no, returns `None` — the test then fails with `"num_layers override requested but ModelLoader does not support it"`

### Running reduced-layer tests

```bash
# LLM benchmark tests
source venv/activate
pytest -svv tests/benchmark/test_llms.py::test_<model_name> --num-layers 1 2>&1 | tee /tmp/bringup_1layer.log

# Encoder benchmark tests
pytest -svv tests/benchmark/test_encoders.py::test_<model_name> --num-layers 1 2>&1 | tee /tmp/bringup_1layer.log
```

### One-layer benchmark runner (batch execution)

The script `tests/benchmark/scripts/run_one_layer_benchmarks.py` runs all benchmark tests with `num_layers=1` automatically. It invokes pytest with `--num-layers 1` for each test and classifies results as `ok`, `failed`, `unsupported`, or `skipped`.

### Full model validation via the runner infra

The runner infra (`tests/runner/test_models.py`) does NOT support `--num-layers` — it always runs the full model. Use it for the final validation step after reduced-layer tests pass:

```bash
pytest -svv tests/runner/test_models.py::test_all_models[<test_id>] 2>&1 | tee /tmp/bringup_full.log
```

## Bringup iteration checklist

1. **`num_layers=1`**: Fix all compilation errors (frontend + tt-mlir). This is the fastest loop.
2. **`num_layers=2`**: Verify multi-layer stacking. Catches KV cache shape issues, layer indexing bugs, or cross-layer state problems.
3. **Full model**: Run with all layers for final accuracy validation (PCC/ATOL). This is the slowest step — only do it after 1-2 layer runs pass.
4. **Record results**: Note the PCC from the full run to set `required_pcc` in the YAML config.
