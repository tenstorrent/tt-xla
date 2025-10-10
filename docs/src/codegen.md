# Code Generation in TT-XLA

This document describes the code generation feature in TT-XLA, which enables you to export JAX or PyTorch models as standalone Python or C++ source code targeting Tenstorrent hardware.

## Overview

Code generation transforms your model into human-readable Python or C++ source code that directly calls the TT-NN library. This allows you to:

* Extract models from JAX or PyTorch into standalone, inspectable code
* Understand and modify the low-level operations your model performs
* Create distributable implementations that don't require the full TT-XLA runtime

We call this part of the TT-Forge compiler: TT-Alchemist.
Or sometimes EmitC or EmitPy internally.

## Usage

Code generation is configured through compile options that you set before compiling your model. You can target either Python or C++ code generation.

## Configuration Options

| Option | Description | Valid Values |
|--------|-------------|--------------|
| `backend` | Target code generation language | `"codegen_py"` (Python) or `"codegen_cpp"` (C++)  |
| `export_path` | Directory where generated code will be written | Any valid directory path (created if doesn't exist) |

### PyTorch

Configure code generation options before compiling your model:

```python
import torch
import torch.nn as nn
import torch_xla
import torch_xla.core.xla_model as xm
import torch_xla.runtime as xr

xr.set_device_type("TT")

class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.A = nn.Linear(32, 128)
        self.B = nn.Linear(128, 64)

    def forward(self, x):
        x = self.A(x)
        x = torch.relu(x)
        x = self.B(x)
        x = torch.tanh(x)
        return torch.sum(x**2)

# Set code generation options
options = {
    "backend": "codegen_py",  # Use "codegen_cpp" for C++ generation
    "export_path": "torch_codegen_example",
}
torch_xla.set_custom_compile_options(options)

device = xm.xla_device()
model = Model()
model.compile(backend="tt")
model = model.to(device)
x = torch.randn(32, 32).to(device)

output = model(x)
```

After running this, check the `torch_codegen_example` directory for:
* `ttir.mlir` - The TTIR intermediate representation
* Generated Python (`.py`) or C++ (`.cpp`, `.h`) source files

### JAX

For JAX, pass compile options directly to `jax.jit()`:

```python
import flax.nnx as nnx
import jax
import jax.numpy as jnp

class Model(nnx.Module):
    def __init__(self, *, rngs: nnx.Rngs):
        self.A = nnx.Linear(32, 128, rngs=rngs)
        self.B = nnx.Linear(128, 64, rngs=rngs)

    def __call__(self, x):
        x = self.A(x)
        x = self.B(x)
        x = nnx.tanh(x)
        return jnp.sum(x**2)

# Initialize model on CPU
with jax.default_device(jax.devices("cpu")[0]):
    model = Model(rngs=nnx.Rngs(0))
    key = jax.random.key(1)
    x = jax.random.normal(key, (32, 32))
    graphdef, state = nnx.split(model)

def forward(graphdef, state, x):
    model = nnx.merge(graphdef, state)
    return model(x)

# JIT compile with code generation options
fun = jax.jit(
    forward,
    compiler_options={
        "backend": "codegen_py",  # Use "codegen_cpp" for C++ generation
        "export_path": "jax_codegen_example"
    }
)
fun(graphdef, state, x)
```

After running this, check the `jax_codegen_example` directory for:
* `ttir.mlir` - The TTIR intermediate representation
* Generated Python (`.py`) or C++ (`.cpp`, `.h`) source files

## Output Files

When code generation completes, your specified output directory will contain:

### TTIR File (`ttir.mlir`)
The Tenstorrent Intermediate Representation of your model - a high-level representation showing the model after initial compilation from the source framework.

### Generated Source Code
* **Python generation** (`codegen_py`): Python (`.py`) files that implement your model using TT-NN API calls.
* **C++ generation** (`codegen_cpp`): C++ source (`.cpp`) and header (`.h`) files for integration into C++ applications.

The generated source code is standalone and can be executed independently of TT-XLA, as long as the TT-NN library is available.

## Use Cases

### Model Portability
Extract models from JAX or PyTorch into standalone code that can be deployed without requiring the full framework stack.

### C++ Integration
Generate C++ implementations for integration into high-performance applications, embedded systems, or existing C++ codebases.

### Model Inspection and Debugging
Examine the generated source code to understand exactly what operations are performed, enabling easier debugging and optimization.

### Educational Purposes
Study the generated code to learn how high-level JAX or PyTorch operations translate to TT-NN library calls.

### Customization
Modify the generated code to add custom optimizations or integrate with existing infrastructure.

## Expected Behavior

When code generation is enabled:

1. The model goes through the compilation pipeline as normal
2. Generated code is written to the specified `export_path` directory
3. **The process may terminate after code generation**. This is expected behavior and indicates successful code generation.

> **NOTE:** The process termination is intentional - code generation is typically a final step to extract your model for deployment or analysis.

## Limitations and Considerations

* **TT-Alchemist Required**: Code generation requires the TT-Alchemist library to be installed in your Python environment
* **Process Termination**: The process may exit after generating code (this is expected)
* **Export Path Required**: You must provide an `export_path` when using `codegen_py` or `codegen_cpp` backends
* **Generated Code Dependencies**: The generated code requires the TT-NN library to execute

## Troubleshooting

### TT-Alchemist Not Found

**Error:**
```
WARNING: tt-alchemist library not found in Python environment
```

**Solution:** Ensure TT-Alchemist is installed in your active Python virtual environment. Check that the `VIRTUAL_ENV` environment variable is set correctly.

### Code Generation Fails

**Error:**
```
ERROR: tt-alchemist generatePython failed
```

**Solutions:**
1. Verify the `export_path` directory is writable
2. Check that `ttir.mlir` was created in the output directory
3. Ensure TT-Alchemist is properly initialized

### Export Path Not Set

**Error:**
```
Compile option 'export_path' must be provided when backend is not 'TTNNFlatbuffer'
```

**Solution:** Add the `export_path` option to your compiler options:
```python
options = {
    "backend": "codegen_py",
    "export_path": "output_directory"  # Add this
}
```

## Examples

Complete working examples are available in the `examples` directory:

* PyTorch: `examples/pytorch/codegen_via_options_example.py`
* JAX: `examples/jax/codegen_via_options_example.py`

## Related Documentation

* [Code Generation Quick Start](codegen_quickstart.md) - Quick introduction for PyTorch
* [Getting Started](getting_started.md) - Main setup guide for TT-XLA
* [Building from Source](getting_started_build_from_source.md) - Development setup instructions

## Where to Go Next

Now that you understand code generation in TT-XLA:
* Experiment with both Python and C++ backends
* Try code generation on your own models
* Examine the generated code to understand how your models are implemented
* Try tweaking a couple of operations and see if you can get the same results a different way

> **NOTE:** If you encounter issues or have questions, please visit the [TT-XLA Issues](https://github.com/tenstorrent/tt-xla/issues) page for assistance.

---

## Alternative: Code Generation via Serialization

> **NOTE:** Users should strongly prefer to go trough compile options. This is documented for completeness sake.

We also allow users to hook into serialization infrastructure and run alchemist directly on the results.

**Examples:**
* PyTorch: `examples/pytorch/codegen_via_serialize_example.py`
* JAX: `examples/jax/codegen_via_serialize_example.py`
