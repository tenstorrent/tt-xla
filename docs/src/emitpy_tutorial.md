# Tutorial: Generate Python Code from Your Model

Learn how to convert PyTorch and JAX models into standalone Python code using TT-XLA's code generation feature.

## What You'll Learn

By the end of this tutorial, you'll be able to:
- Generate Python code from PyTorch/JAX models using TT-XLA
- Execute the generated code on Tenstorrent hardware
- Inspect and understand the TT-NN API calls in the generated code

> **Time to complete:** ~15 minutes

## What is Code Generation?

Code generation (also called "EmitPy" or powered by "TT-Alchemist") transforms your high-level model into human-readable Python source code that directly calls the TT-NN library. This lets you inspect, modify, and deploy models without the full TT-XLA runtime.

For complete conceptual overview and all options, see the [Code Generation Guide](./getting_started_codegen.md).

---

## Prerequisites

Before starting, ensure you have:

- [ ] Access to Tenstorrent hardware (via IRD or physical device) - jump to [Step 1](#step-1-reserve-hardware-and-start-docker-container)
- [ ] TT-XLA Docker image: `ghcr.io/tenstorrent/tt-xla/tt-xla-ird-ubuntu-22-04:latest` - jump to [Step 2](#step-2-clone-and-setup-tt-xla)

---

## Step-by-Step Guide

### Step 1: Reserve Hardware and Start Docker Container

Reserve Tenstorrent hardware with the TT-XLA Docker image:

```bash
ird reserve --docker-image ghcr.io/tenstorrent/tt-xla/tt-xla-ird-ubuntu-22-04:latest [additional ird options]
```

> **Tip:** The `[additional ird options]` should include your typical IRD configuration like architecture, number of chips, etc.

### Step 2: Clone and Setup TT-XLA

Inside your Docker container, clone the repository:

```bash
git clone https://github.com/tenstorrent/tt-xla.git
cd tt-xla
```

Initialize submodules (required for dependencies):

```bash
git submodule update --init --recursive
```

**Expected output:** Git will download all third-party dependencies.

### Step 3: Build TT-XLA

Set up the build environment and compile the project:

```bash
# Activate the Python virtual environment
source venv/activate

# Configure the build
cmake -G Ninja -B build

# Build the project (this may take 10-15 minutes)
cmake --build build
```

> **Debug Build:** Add `-DCMAKE_BUILD_TYPE=Debug` to the cmake configure command if you need debug symbols.

### Step 4: Run Code Generation Example

Choose your framework and run the example:

**PyTorch:**
```bash
python examples/pytorch/codegen/custom_module.py
```

**JAX:**
```bash
python examples/jax/codegen/custom_module.py
```

#### What Happens During Code Generation

Both examples configure TT-XLA with these options:

```python
options = {
    # Code generation options
    "backend": "codegen_py",
    # Export path
    "export_path": "model",
}
```

The process will:
1. ✅ Compile your model through the TT-XLA pipeline
2. ✅ Generate Python source code in the `model/` directory

#### Generated Files

Check the `model/` directory for your generated code:

```bash
ls -la model/
```

You should see:
- **`main.py`** - Generated Python code with TT-NN API calls
- **`run`** - Executable shell script to run the generated code
- **`tensors/`** - Directory with exported model input and parameter tensors
- **`irs/`** - # VHLO, SHLO, TTIR, TTNN intermediate representations (debugging)

### Step 5: Generate the optimized code
We can specify different optimization options in order to produce the more performant code. For example, we can supply following set of options to produce the optimized code.
```python
options = {
    # Code generation options
    "backend": "codegen_py",

    # Optimizer options
    "enable_optimizer": True,
    "enable_memory_layout_analysis": True,
    "enable_l1_interleaved": False,

    # Export path
    "export_path": "model",
}
```

Link to other optimizer options to be added here. [\[#1849\] TT-XLA Optimizer Docs](https://github.com/tenstorrent/tt-xla/issues/1849)

### Step 6: Exporting model input and parameter tensors

By default, model input and parameter tensors are exported to `export_path/tensors/`.

If you don't need to dump these tensors, set the compile option `"export_tensor": False`. The generated code will use `ttnn.ones` for input and parameter tensors instead.

```python
options = {
    # Code generation options
    "backend": "codegen_py",
    # Export path
    "export_path": "model",
    "export_tensors": False
}
```

### Step 7: Execute the Generated Code

Navigate to the model directory and run the execution script:

```bash
cd model
./run
```

#### What the `run` Script Does

The script automatically:
- Sets up the Python environment with TT-NN dependencies
- Configures Tenstorrent hardware settings
- Executes the generated Python code
- Displays inference results

**Expected output:** You should see inference results printed to the console, showing your model running successfully on Tenstorrent hardware.

---

## Next Steps

Now that you've successfully generated and executed code:

### Inspect the Generated Code

Open `model/main.py` to see how your PyTorch/JAX operations map to TT-NN API calls:

```bash
cat model/main.py
```

Look for patterns like:
- Tensor allocation and initialization
- Operation implementations (matrix multiply, activation functions, etc.)
- Memory management and device synchronization

### Customize the Generated Code

Try modifying operations in `main.py`:
- Change tensor shapes or data types
- Add print statements to debug intermediate values
- Optimize memory layouts or operation ordering

### Generate C++ Code

Want C++ instead of Python? Change the backend:

```python
options = {
    "backend": "codegen_cpp",  # Generate C++ code
    "export_path": "model_cpp",
}
```

The generated C++ code is **fully standalone** and can be integrated into existing C++ projects.

### Generate resnet TTNN code using following example:
- [PyTorch Resnet example](../../examples/pytorch/codegen/resnet.py)
- [Jax Resnet example](../../examples/jax/codegen/resnet.py)

### Learn More

- **[Code Generation Guide](./getting_started_codegen.md)** - Complete reference for all options and use cases
- **[PyTorch Example Source](../../examples/pytorch/codegen/custom_module.py)** - Full example code
- **[JAX Example Source](../../examples/jax/codegen/custom_module.py)** - Full example code

---

## Summary

**What you accomplished:**
- ✅ Built TT-XLA from source in Docker
- ✅ Generated Python code from a PyTorch/JAX model
- ✅ Executed the generated code on Tenstorrent hardware
- ✅ Learned where to find and inspect the generated code

**Key takeaways:**
- Code generation creates inspectable implementations
- The process intentionally terminates after generation (current limitation)
- Generated code can be modified and optimized for your use case

---

**Need help?** Visit the [TT-XLA Issues](https://github.com/tenstorrent/tt-xla/issues) page or check the [Code Generation Guide](./getting_started_codegen.md) for more details.
