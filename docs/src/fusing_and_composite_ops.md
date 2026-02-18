# Fusing and Composite Ops

When PyTorch models are compiled through `torch.compile("tt")`, high-level operations like `RMSNorm` or `GELU` are typically decomposed by XLA into many primitive ops. TT-XLA addresses this with two different mechanisms:

- **Composite Ops**: a StableHLO-level mechanism that gives us option to wrap high-level ops (for example `tenstorrent.rms_norm`) and preserve them as single ops in TT-MLIR.
- **Torch FX Fusing**: a graph-rewrite mechanism that pattern-matches multi-op FX subgraphs and rewrites them into standard torch ops (for example `torch.nn.functional.rms_norm`).

These mechanisms are different, but they are designed to work together. In practice, fusing is only useful because composites exist: fusion rewrites user code into composite-eligible ops, and composites are what preserve that intent through decomposition so TT-MLIR can lower it to optimized TTNN operations. There is also an advanced MLIR-level fusing system in the `tt-mlir` repo, covered briefly at the end.

## Compilation Pipeline Overview

The following diagram shows where fusing and composite ops fit in the compilation pipeline:

```
PyTorch Model
  |
  v
Torch compilation
  |
  v
FX Graph (torch.fx.GraphModule)
  |
  v
run_fusion_passes()          <-- Torch FX Fusing
  |                               Detects multi-op patterns (e.g. LlamaRMSNorm)
  |                               and replaces them with standard torch ops
  v                               (e.g. torch.nn.functional.rms_norm)
handle_composite_ops()       <-- Composite Wrapping
  |                               Wraps known torch ops with StableHLO
  |                               composite markers (e.g. tenstorrent.rms_norm)
  v
torch.export + torch decompositions  <-- Wrapped composites survive decomposition
  |
  v
torch to hlo conversions  <-- Wrapped composites survive decomposition
  |
  v
StableHLO
  |
  v
TTIR legalization     <-- TT-MLIR recognizes wrapped composites
  |
  v
TTNN -> Hardware
```

## Configuration Options

Both Torch FX Fusing and Composite Ops can be toggled via `torch.compile` options:

| Option | Default | Description |
|--------|---------|-------------|
| `tt_enable_torch_fx_fusion_pass` | `True` | Enable/disable Torch FX fusion pattern matching |
| `tt_enable_composite_ops` | `True` | Enable/disable composite op wrapping |

Example usage:

```python
import torch

model = MyModel()
input = torch.randn(1, 32, 768)

# Enable both (default)
compiled = torch.compile(model, backend="tt")

# Disable fusion, keep composites
compiled = torch.compile(model, backend="tt", options={
    "tt_enable_torch_fx_fusion_pass": False,
    "tt_enable_composite_ops": True,
})

# Disable both (useful for debugging)
compiled = torch.compile(model, backend="tt", options={
    "tt_enable_torch_fx_fusion_pass": False,
    "tt_enable_composite_ops": False,
})
```

## Fusion + Composites: Working Together

The two systems are designed to chain together. Fusion converts arbitrary user implementations into *standard* torch ops, and composites wrap those standard ops for the compiler.

Here is a concrete walkthrough using LlamaRMSNorm:

```
Step 1: User's LlamaRMSNorm model code
  hidden_states = hidden_states.to(float32)
  variance = hidden_states.pow(2).mean(-1, keepdim=True)
  hidden_states = hidden_states * torch.rsqrt(variance + eps)
  return weight * hidden_states.to(input_dtype)

Step 2: run_fusion_passes() — RMSNormFusionProvider matches this pattern
  → Replaced with: torch.nn.functional.rms_norm(hidden_states, weight.shape, weight, eps)

Step 3: handle_composite_ops() — rms_norm is in the replacements dict
  → Wrapped as: composite_rms_norm(hidden_states, weight.shape, weight, eps)
  → In the FX graph, this creates StableHLO composite markers around rms_norm

Step 4: torch.export + torch decompositions
  → Wrapped composites survive decomposition as "tenstorrent.rms_norm"

Step 5: torch to hlo conversions
  → Wrapped composites survive decomposition as "tenstorrent.rms_norm"

Step 6: TTIR legalization
  → Recognized and lowered to optimized TTIR rms_norm op
  → Compiled to TTNN and executed on hardware
```

Without fusion, users who write their own RMSNorm implementation rather than calling `torch.nn.functional.rms_norm` directly (e.g. LlamaRMSNorm in huggingface transformers), would not benefit from the composite optimization. The fusion pass bridges this gap.

## Composite Ops

### What Are Composite Ops

StableHLO composite ops are a mechanism for wrapping a sequence of operations and giving them a *name* that custom backends can recognize.

TT-XLA uses the naming convention `tenstorrent.<op_name>` (e.g., `tenstorrent.gelu`, `tenstorrent.rms_norm`, `tenstorrent.layer_norm`). When these composites reach TT-MLIR, the `LegalizeStableHLOCompositeToTTIR` pass recognizes them and maps them to optimized TTIR operations.

### How They Work

Each composite op follows a 3-step pattern using `StableHLOCompositeBuilder`:

1. **Mark inputs** — call `builder.mark_inputs(...)` on the input tensors
2. **Run the original op** — call the standard torch op
3. **Mark outputs** — call `builder.mark_outputs(...)` on the result

Here is `composite_gelu` example. [View full source](https://github.com/tenstorrent/tt-xla/blob/main/python_package/tt_torch/composite_ops.py)

```python
{{#include ../../../python_package/tt_torch/composite_ops.py:30:47}}
```

The `name` parameter becomes the composite name in StableHLO (e.g., `tenstorrent.gelu`). The `attr` dictionary passes metadata attributes to the compiler (e.g., epsilon value).

### The Replacements Dictionary

The `replacements` dictionary in `composite_ops.py` maps torch functions and module types to their composite implementations:

[View full source](https://github.com/tenstorrent/tt-xla/blob/main/python_package/tt_torch/composite_ops.py)

```python
{{#include ../../../python_package/tt_torch/composite_ops.py:194:202}}
```

The `handle_composite_ops` pass iterates over the FX graph and uses this dictionary:

[View full source](https://github.com/tenstorrent/tt-xla/blob/main/python_package/tt_torch/backend/passes.py)

```python
{{#include ../../../python_package/tt_torch/backend/passes.py:32:56}}
```

There are two replacement categories:
- **Function replacements** (`call_function` nodes): The node's `target` is swapped directly from `torch.nn.functional.gelu` to `composite_gelu`.
- **Module replacements** (`call_module` nodes): A replacement function (e.g., `replace_layer_norm_module`) creates new `get_attr` nodes for the module's parameters and replaces the `call_module` node with a `call_function` node targeting the composite function.

### How to Add a New Composite Op

1. **Define the composite function** in `python_package/tt_torch/composite_ops.py` using `StableHLOCompositeBuilder`:
   ```python
   def composite_my_op(input: Tensor, param: float) -> Tensor:
       attr = {"param": param}
       builder = StableHLOCompositeBuilder(name="tenstorrent.my_op", attr=attr)

       input = builder.mark_inputs(input)
       output = torch.nn.functional.my_op(input, param)
       output = builder.mark_outputs(output)
       return output
   ```

2. **Add to the `replacements` dictionary**:
   ```python
   replacements = {
       ...
       torch.nn.functional.my_op: composite_my_op,
   }
   ```

3. **For `nn.Module` types**, write a `replace_<op>_module` function that:
   - Extracts parameters from the module instance
   - Creates `get_attr` nodes for module weights/biases
   - Replaces the `call_module` node with a `call_function` node
   - See `replace_layer_norm_module` in [composite_ops.py](https://github.com/tenstorrent/tt-xla/blob/main/python_package/tt_torch/composite_ops.py) for a complete example.

4. **Write tests** in `tests/torch/ops/test_composite_ops.py`:
   ```python
   @pytest.mark.single_device
   def test_patched_my_op(request):
       class MyModel(torch.nn.Module):
           def forward(self, x):
               return torch.nn.functional.my_op(x, param=0.5)

       options = {"tt_enable_composite_ops": True}
       input = torch.randn(32, 32)
       run_graph_test(
           MyModel(), [input],
           comparison_config=ComparisonConfig(),
           framework=Framework.TORCH,
           torch_options=options,
       )
   ```

5. **Ensure TT-MLIR has a handler** for the composite name (`tenstorrent.my_op`). The composite will only be lowered to an optimized implementation if the `StableHLOLegalizeCompositePass` in TT-MLIR recognizes it.

## Torch FX Fusing

### How It Works

Torch FX fusing uses PyTorch's `replace_pattern_with_filters` API, which performs *subgraph isomorphism matching* on the FX graph. You define two functions:

- **`pattern`**: A function that constructs the subgraph you want to find. When traced, it becomes a template that the matcher searches for in the model's FX graph.
- **`replacement`**: A function with the same signature that constructs the replacement subgraph.

The matcher finds all occurrences of the pattern subgraph and substitutes them with the replacement. An optional **`match_filter`** function can inspect each match and decide whether to accept or reject it (e.g., based on tensor shapes or hardware constraints).

### The FusionProvider Framework

All fusion providers inherit from the `FusionProvider` base class.

[View full source](https://github.com/tenstorrent/tt-xla/blob/main/python_package/tt_torch/fusion_providers.py)

```python
class FusionProvider(ABC):
    """Base class for all fusion pattern providers.
    Subclasses are automatically registered via __init_subclass__."""

    _registered_providers: List[Type["FusionProvider"]] = []

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        FusionProvider._registered_providers.append(cls)

    @property
    @abstractmethod
    def name(self) -> str: ...

    @staticmethod
    @abstractmethod
    def pattern(*args, **kwargs) -> Tensor: ...

    @staticmethod
    @abstractmethod
    def replacement(*args, **kwargs) -> Tensor: ...
```

Key points:
- `_registered_providers` collects all subclasses automatically via `__init_subclass__`
- Subclasses must implement `name`, `pattern`, and `replacement`
- Override `match_filter` for a single filter, or `get_match_filters` for multiple filters
- `replace_pattern()` (see full source) calls `replace_pattern_with_filters` with the provider's pattern, replacement, and filters

The `run_fusion_passes` function in `passes.py` iterates over all registered providers and applies them:

[View full source](https://github.com/tenstorrent/tt-xla/blob/main/python_package/tt_torch/backend/passes.py)

```python
{{#include ../../../python_package/tt_torch/backend/passes.py:11:29}}
```

### Example: RMSNormFusionProvider

The `RMSNormFusionProvider` detects the common LlamaRMSNorm pattern and replaces it with `torch.nn.functional.rms_norm`. [View full source](https://github.com/tenstorrent/tt-xla/blob/main/python_package/tt_torch/fusion_providers.py)

The `pattern` and `replacement` methods define what to match and what to substitute:

```python
{{#include ../../../python_package/tt_torch/fusion_providers.py:115:140}}
```

Notable details:
- **`.add()`/`.mul()` instead of `+`/`*`**: Dynamo traces tensor operations as `call_method` nodes, not `call_function`. The pattern must match the traced form.
- **`dtype` parameter as wildcard**: Including `dtype` as a parameter makes it match *any* value in that position, so the pattern works regardless of the cast target dtype.

The optional `match_filter` inspects each match and can reject it based on hardware constraints:

```python
{{#include ../../../python_package/tt_torch/fusion_providers.py:142:165}}
```

This filter uses `node.meta["example_value"]` to inspect tensor shapes at match time, skipping fusion when the weight dimension exceeds what the hardware currently supports.

### How to Add a New Fusion Pattern

1. **Identify the pattern in the FX graph.** Use `torch.compile` with a print/debug backend, or call `gm.print_readable()` to inspect the traced graph as readable Python code and find the multi-op sequence you want to fuse.

2. **Create a `FusionProvider` subclass** in `python_package/tt_torch/fusion_providers.py`:
   ```python
   class MyOpFusionProvider(FusionProvider):
       @property
       def name(self) -> str:
           return "my_op_fusion"

       @staticmethod
       def pattern(x: Tensor, ...) -> Tensor:
           # Reproduce the exact sequence of ops from the FX graph
           ...

       @staticmethod
       def replacement(x: Tensor, ...) -> Tensor:
           # Replace with a single torch op
           ...
   ```

3. **Implement `pattern`**: Write a function that reproduces the exact subgraph you want to match. Use `.add()`, `.mul()`, etc. instead of operators. Parameters that should match any value act as wildcards.

4. **Implement `replacement`**: Write a function with the same signature that produces the desired replacement. This is typically a single torch op like `torch.nn.functional.rms_norm`.

5. **Optionally implement `match_filter`**: If the pattern should only match under certain conditions (tensor shapes, dtypes, etc.), override `match_filter` to inspect `match.nodes_map` and return `False` for invalid matches.

6. **Write a test** in `tests/torch/ops/test_fusion_ops.py`:
   ```python
   @pytest.mark.single_device
   @pytest.mark.push
   def test_my_op_fusion(request):
       options = {
           "tt_enable_torch_fx_fusion_pass": True,
           "tt_enable_composite_ops": True,
       }
       model = MyModel()
       input_tensor = torch.randn(1, 32, 32)
       run_graph_test(
           model, [input_tensor],
           comparison_config=ComparisonConfig(),
           framework=Framework.TORCH,
           torch_options=options,
           request=request,
       )
   ```

### Tips and Pitfalls

- **Use method calls, not operators.** In the pattern function, always use `.add()`, `.mul()`, `.sub()`, `.div()` instead of `+`, `*`, `-`, `/`. Dynamo traces these differently.
- **Fusion runs before composites.** The pipeline runs fusion first, then composite wrapping. This means your fused replacement op (e.g., `rms_norm`) can then be picked up by the composite system.
- **Test with and without fusion.** Verify your fusion produces numerically correct results by comparing against the unfused model.
- **Inspect the FX graph.** To debug pattern matching issues, call `gm.print_readable()` before and after `run_fusion_passes()` in the pipeline. This outputs the graph as readable Python code (see [PyTorch docs](https://docs.pytorch.org/docs/stable/fx.html)).

## MLIR Fusing (Advanced)

TT-MLIR also supports fusing at the MLIR level, as an alternative to the Torch FX + Composites approach described above. The two approaches have different trade-offs:

| | Torch FX + Composites | MLIR Fusing |
|---|---|---|
| **Advantages** | Easier to write and debug (Python-based pattern matching), lower barrier to entry  | Agreed-upon best location for fusions to live long-term. Has better context about hardware-specific optimizations |
| **Limitations** | All torch-fused operations must be wrapped inside a composite op and legalized in tt-mlir to prevent decomposition during torch\_xla lowering | Requires MLIR pattern matching syntax, which is harder to write and debug. Higher barrier to entry for new contributors |

In addition to the Torch FX level fusing described above, TT-MLIR has its own pattern matching and fusion passes at the MLIR level. These operate on the TTIR and TTNN dialects *after* StableHLO conversion.

Key MLIR fusing components (in the [tt-mlir repository](https://github.com/tenstorrent/tt-mlir)):
- **Canonicalizers**: Simplify and normalize MLIR operations (e.g., folding constants, simplifying identity ops)
- **TTIRFusing**: Fuses patterns at the TTIR dialect level
- **TTNNFusing**: Fuses patterns at the TTNN dialect level, closer to hardware
- **Pattern rewriters**: Use the MLIR `PatternRewriter` infrastructure for subgraph matching and replacement

For more on MLIR pattern rewriting, see the [MLIR Pattern Rewriter documentation](https://mlir.llvm.org/docs/PatternRewriter/).
