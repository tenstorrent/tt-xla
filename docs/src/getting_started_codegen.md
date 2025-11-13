# Code Generation Guide

Convert JAX or PyTorch models into standalone Python or C++ source code targeting Tenstorrent hardware.

---

## Quick Reference

| Framework | Backend Option | Output | Standalone? |
|-----------|----------------|--------|-------------|
| PyTorch/JAX | `codegen_py` | Python (`.py`) | No (requires TT-XLA build) |
| PyTorch/JAX | `codegen_cpp` | C++ (`.cpp`, `.h`) | Yes |

**New to code generation?** Start with the [Python Code Generation Tutorial](./emitpy_tutorial.md) for a hands-on walkthrough.

---

## Overview

Code generation (powered by **TT-Alchemist**) transforms your model into human-readable source code that directly calls the TT-NN library, enabling:

- **Customization** - Modify generated code to add optimizations or integrate with existing infrastructure
- **Model Portability** - Extract models into standalone code deployable without the full framework stack
- **Inspection & Debugging** - Examine generated source to understand exact operations performed
- **Education** - Study how high-level framework operations translate to TT-NN library calls

> **Technical Note:** Internally referred to as TT-Alchemist, EmitPy (Python generation), or EmitC (C++ generation).

---

## Basic Usage

### PyTorch

Configure code generation options before compiling your model:

```python
import torch
import torch_xla.core.xla_model as xm

# Configure code generation
options = {
    "backend": "codegen_py",              # Or "codegen_cpp" for C++
    "export_path": "torch_codegen_output" # Output directory
}
torch_xla.set_custom_compile_options(options)

# Standard PyTorch workflow
device = xm.xla_device()
model = YourModel()
model.compile(backend="tt")
model = model.to(device)
x = torch.randn(32, 32).to(device)

# Trigger code generation
output = model(x)
```

**Output location:** `torch_codegen_output/` directory containing:
- `ttir.mlir` - TTIR intermediate representation
- `*.py` or `*.cpp`/`*.h` - Generated source files

### JAX

Pass compile options directly to `jax.jit()`:

```python
import jax
from flax import nnx

def forward(graphdef, state, x):
    model = nnx.merge(graphdef, state)
    return model(x)

# JIT compile with code generation
jitted_forward = jax.jit(
    forward,
    compiler_options={
        "backend": "codegen_py",          # Or "codegen_cpp" for C++
        "export_path": "jax_codegen_output"
    }
)

# Trigger code generation
result = jitted_forward(graphdef, state, x)
```

**Output location:** `jax_codegen_output/` directory containing:
- `ttir.mlir` - TTIR intermediate representation
- `*.py` or `*.cpp`/`*.h` - Generated source files

---

## Configuration Options

### Codegen Options

| Option | Type | Description |
|--------|------|-------------|
| `backend` | `string` | Code generation target:<br>• `"codegen_py"` - Generate Python code<br>• `"codegen_cpp"` - Generate C++ code |
| `export_path` | `string` | Directory for generated code (created if doesn't exist) |
| `export_tensors` | `bool` | Whether to dump model input and parameter tensors to disk (**False** by default) |

### Example Configurations

**Python Generation:**
```python
options = {
    "backend": "codegen_py",
    "export_path": "./generated_python"
    "export_tensors": True
}
```

**C++ Generation:**
```python
options = {
    "backend": "codegen_cpp",
    "export_path": "./generated_cpp"
    #export_tensors -> default True when doing codegen
}
```

---

## Generated Output

### Directory Structure

After code generation completes, your `export_path` directory contains:

```
<export_path>/
├── ttir.mlir          # TTIR intermediate representation (debugging)
├── main.py/cpp        # Generated Python/C++ code
├── run                # Execution script
└── tensors/     # Directory with dumped tensors if specified by export_tensors option
```

### File Descriptions

**`ttir.mlir`** - Tenstorrent Intermediate Representation
- High-level representation after initial compilation
- Useful for debugging compilation issues
- Human-readable MLIR dialect

**Generated Python (`*.py`)** - Python Implementation
- Direct TT-NN API calls
- Human-readable and modifiable
- **Not standalone** - requires TT-XLA build to execute
- Includes `run` script for execution

**Generated C++ (`*.cpp`, `*.h`)** - C++ Implementation
- Direct TT-NN API calls
- Human-readable and modifiable
- **Fully standalone** - only requires TT-NN library
- Can be integrated into existing C++ projects

**tensors/** - Serialized model inputs and parameters (created when `export_tensors: True`)
- Used by the generated code to load real model inputs and weights instead of random values

---

## Code Generation Behavior

### Expected Process Flow

1. ✅ Model compiles through TT-XLA pipeline
2. ✅ Code generation writes files to `export_path`
3. ⚠️ **Process may terminate with error** (expected behavior)

### Process Termination

> **Important:** The process terminating after code generation is **expected behavior** when running through the frontend.

**You'll see this error message:**
```
Standalone solution was successfully generated. Executing codegen through the frontend is not supported yet.
Unfortunately your program will now crash :(
ERROR:root:Caught an exception when exiting the process.
RuntimeError: Bad StatusOr access: UNIMPLEMENTED: Error code: 12
```

**This is normal!** Your code was generated successfully. The error simply indicates that continuing execution through the frontend isn't currently supported.

### Verifying Success

Check that code generation succeeded:

```bash
ls -la <export_path>/
```

You should see:
- ✅ `ttir.mlir` file exists
- ✅ Generated source files (`.py` or `.cpp`/`.h`)
- ✅ File sizes are non-zero
- ✅ (Python only) Executable `run` script exists

---

## Use Cases

### 1. Custom Optimization

**Scenario:** Hand-tune generated code for specific workloads

**Benefits:**
- Modify operation ordering
- Adjust memory layouts
- Add custom kernel calls

**Best for:** Performance-critical applications, specialized hardware configurations

### 2. Model Deployment & Portability

**Scenario:** Deploy models without the full JAX/PyTorch stack

**Benefits:**
- Smaller deployment footprint
- Fewer dependencies
- Direct control over execution

**Best for:** Production environments, edge devices, embedded systems

### 3. Model Inspection & Debugging

**Scenario:** Understand what operations your model performs

**Benefits:**
- Examine exact TT-NN API calls
- Identify performance bottlenecks
- Understand memory access patterns

**Best for:** Performance optimization, debugging accuracy issues

### 4. Educational & Research

**Scenario:** Learn how high-level operations translate to hardware

**Benefits:**
- Study framework→hardware mapping
- Experiment with low-level optimizations
- Understand compiler transformations

**Best for:** Learning, research, optimization experiments

---

## Advanced Topics

### Alternative: Code Generation via Serialization

> **Note:** Most users should use compile options (documented above). This method is provided for advanced use cases.

You can also invoke code generation by hooking into the serialization infrastructure and running TT-Alchemist directly on the results.

**When to use this:**
- Custom compilation workflows
- Integration with existing build systems
- Automated pipeline generation

**Examples:**
- PyTorch: [`examples/pytorch/custom_module.py`](../../examples/pytorch/codegen/custom_module.py
- JAX: [`examples/jax/custom_module.py`](../../examples/jax/codegen/custom_module.py)

---

## Related Documentation

- **[Python Code Generation Tutorial](./emitpy_tutorial.md)** - Step-by-step hands-on tutorial
- **[Getting Started Guide](./getting_started.md)** - Main TT-XLA setup
- **[Building from Source](./getting_started_build_from_source.md)** - Development setup

---

## Next Steps

1. **Try the tutorial:** Follow the [Python Code Generation Tutorial](./emitpy_tutorial.md) for hands-on experience
2. **Experiment:** Try both `codegen_py` and `codegen_cpp` backends
3. **Inspect code:** Examine generated code to understand TT-NN API usage
4. **Customize:** Modify generated code to optimize for your use case
5. **Deploy:** Integrate generated C++ code into your applications

---

**Questions or issues?** Visit [TT-XLA GitHub Issues](https://github.com/tenstorrent/tt-xla/issues) for support.
