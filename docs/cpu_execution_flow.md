# CPU Execution Flow — End-to-End Documentation

This document traces the complete CPU execution path in the tt-xla test infrastructure, from tester initialization through compilation, device transfer, execution, and result return.

---

## Table of Contents

1. [Overview](#1-overview)
2. [Phase 1 — Initialization](#2-phase-1--initialization)
   - [BaseTester Setup](#21-basetester-setup)
   - [ModelTester Component Initialization](#22-modeltester-component-initialization)
   - [TorchModelTester Specifics](#23-torchmodeltester-specifics)
3. [Phase 2 — Compilation (`torch.compile`)](#3-phase-2--compilation-torchcompile)
   - [What `compile_torch_workload_for_cpu` Does](#31-what-compile_torch_workload_for_cpu-does)
   - [What `torch.compile` Returns](#32-what-torchcompile-returns)
4. [Phase 3 — CPU Execution (`_run_on_cpu`)](#4-phase-3--cpu-execution-_run_on_cpu)
   - [JAX Accelerator Masking](#41-jax-accelerator-masking)
   - [Device Connection](#42-device-connection)
   - [Moving Workload to CPU](#43-moving-workload-to-cpu)
   - [Actual Execution](#44-actual-execution)
5. [Phase 4 — Inductor Internals (First Run)](#5-phase-4--inductor-internals-first-run)
   - [TorchDynamo — Bytecode Tracing](#51-torchdynamo--bytecode-tracing)
   - [Inductor Frontend — Graph Optimizations](#52-inductor-frontend--graph-optimizations)
   - [Inductor Backend — CPU Code Generation](#53-inductor-backend--cpu-code-generation)
   - [Compiled Code Execution](#54-compiled-code-execution)
6. [Phase 5 — Cached Execution (Subsequent Runs)](#6-phase-5--cached-execution-subsequent-runs)
   - [Guard Checking](#61-guard-checking)
   - [Recompilation Triggers](#62-recompilation-triggers)
7. [Result Return Path](#7-result-return-path)
8. [How CPU Results Are Used](#8-how-cpu-results-are-used)
9. [Complete Call Chain Diagram](#9-complete-call-chain-diagram)
10. [Key Source Files](#10-key-source-files)

---

## 1. Overview

The CPU execution flow produces a **golden reference output** — a trusted result computed on the host CPU. This result is later compared against the output from the Tenstorrent (TT) device to verify correctness.

The high-level steps are:

```
Initialize Tester → Build Workload → Compile for CPU → Run on CPU → Return Golden Result
```

The CPU compilation uses PyTorch's `torch.compile` with the **Inductor** backend, which generates optimized C++/OpenMP code for the CPU. Compilation is **lazy** — the actual code generation happens during the first execution, not during the `torch.compile` call itself.

---

## 2. Phase 1 — Initialization

Before any CPU execution happens, the test infrastructure sets up the tester, model, inputs, device runner, and workload.

### 2.1 BaseTester Setup

**File:** `tests/infra/testers/base_tester.py`

When a concrete tester (e.g., `TorchModelTester`) is instantiated, the `BaseTester.__init__` runs first:

```python
class BaseTester(ABC):
    def __init__(self, evaluator_type, comparison_config, framework, ...):
        self._framework = framework            # Framework.TORCH
        self._device_runner = None             # Set below
        self._evaluator = None                 # Set below
        self._initialize_framework_specific_helpers()
```

`_initialize_framework_specific_helpers` does two things:

1. **Creates the DeviceRunner** via `DeviceRunnerFactory.create_runner(Framework.TORCH)`:
   - Creates a `TorchDeviceConnector` singleton which registers the XLA plugin and sets the device type to `"TT"` via `xr.runtime.set_device_type("TT")`.
   - Wraps it in a `TorchDeviceRunner`.

2. **Creates the Evaluator** via `EvaluatorFactory.create_evaluator(...)`:
   - Sets up the comparison evaluator that will later compare CPU vs TT results.

### 2.2 ModelTester Component Initialization

**File:** `tests/infra/testers/single_chip/model/model_tester.py`

`ModelTester.__init__` calls `_initialize_components()`, which runs a 5-step pipeline:

| Step | Method | What It Does |
|------|--------|--------------|
| 1 | `_initialize_model()` | Calls the subclass's `_get_model()` to get the `nn.Module`, then `_configure_model()` to set eval/train mode |
| 2 | `_set_model_dtype()` | If `dtype_override` is set (e.g., `bfloat16`), casts the model's parameters via `model.to(dtype)` |
| 3 | `_cache_model_inputs()` | Calls the subclass's `_get_input_activations()` to generate/load input tensors |
| 4 | `_set_inputs_dtype()` | If `dtype_override` is set, casts float input tensors to the target dtype |
| 5 | `_initialize_workload()` | Packs the model + args + kwargs into a `TorchWorkload` object |

### 2.3 TorchModelTester Specifics

**File:** `tests/infra/testers/single_chip/model/torch_model_tester.py`

The `TorchModelTester` provides Torch-specific overrides:

- **`_configure_model_for_inference`**: Calls `model.eval()` — disables dropout, uses running stats for BatchNorm.
- **`_configure_model_for_training`**: Calls `model.train()` — enables dropout, computes batch stats.
- **`_initialize_workload`**: Creates a `TorchWorkload` with `model`, `args`, `kwargs`, and optional `mesh`/`shard_spec_fn` for multi-chip scenarios.

After initialization completes, the tester holds:
- `self._model` — the `nn.Module`
- `self._workload` — a `TorchWorkload` containing the model, its forward args/kwargs, and a `None` `compiled_executable`
- `self._device_runner` — a `TorchDeviceRunner` ready to execute on CPU or TT

---

## 3. Phase 2 — Compilation (`torch.compile`)

### 3.1 What `compile_torch_workload_for_cpu` Does

**File:** `tests/infra/utilities/utils.py`

```python
def compile_torch_workload_for_cpu(workload: Workload) -> None:
    to_compile = workload.model if workload.model is not None else workload.executable
    workload.compiled_executable = torch.compile(to_compile, backend="inductor")
```

1. **Picks what to compile**: Prefers `workload.model` (an `nn.Module`) over `workload.executable` (a plain callable).
2. **Calls `torch.compile`**: Wraps the callable with the Inductor backend.
3. **Stores the wrapper**: Sets `workload.compiled_executable` to the returned wrapper.

This is called from `TorchModelTester._compile_for_cpu()`.

### 3.2 What `torch.compile` Returns

`torch.compile` does **not** compile anything immediately. It returns a thin lazy wrapper:

| Input Type | Return Type | Description |
|------------|-------------|-------------|
| `nn.Module` | `OptimizedModule` | Wraps the module; intercepts `__call__` to trace and compile on first invocation |
| Plain callable | Decorated callable | Same idea — wraps the function for lazy compilation |

The returned object is **callable with the same signature** as the original — same inputs, same outputs — but internally routes through the Dynamo+Inductor compilation pipeline on the first call.

At this point, `workload.compiled_executable` holds the lazy wrapper. No C++ code has been generated yet.

---

## 4. Phase 3 — CPU Execution (`_run_on_cpu`)

### 4.1 JAX Accelerator Masking

**File:** `tests/infra/testers/single_chip/model/torch_model_tester.py`

```python
def _run_on_cpu(self, compiled_workload: Workload) -> torch.Tensor:
    with _mask_jax_accelerator():
        return super()._run_on_cpu(compiled_workload)
```

**Why this is needed:** When `torch_xla` is imported, it registers `"jax"` as a PyTorch accelerator. During execution, Inductor calls `torch.accelerator.current_device_index()`, which crashes for the JAX accelerator because it doesn't support that API.

**How the masking works** (`_mask_jax_accelerator` context manager):
1. Saves the original `torch.accelerator.is_available` function.
2. Monkey-patches it with a version that returns `False` when the current accelerator is `"jax"`.
3. Inductor now sees no accelerator and generates pure CPU code without issues.
4. On context exit, restores the original function.

### 4.2 Device Connection

The call chain from `_run_on_cpu` down to actual device connection:

```
TorchModelTester._run_on_cpu(workload)
  → ModelTester._run_on_cpu(workload)           # model_tester.py
    → DeviceRunner.run_on_cpu(workload)          # device_runner.py
      → DeviceRunner.run_on_device(workload, DeviceType.CPU)
        → device = self._device_connector.connect_device(DeviceType.CPU, 0)
```

**File:** `tests/infra/connectors/torch_device_connector.py`

For CPU, `TorchDeviceConnector._connect_device` returns:

```python
torch.device("cpu")  # Standard PyTorch CPU device handle
```

No XLA plugin or special discovery is needed for CPU — it's a plain host device.

### 4.3 Moving Workload to CPU

**File:** `tests/infra/runners/torch_device_runner.py`

`DeviceRunner.run_on_device` calls `_put_on_device(workload, device=cpu)`, which delegates to `_safely_put_workload_on_device`:

```python
def _safely_put_workload_on_device(self, workload, device):
    # 1. Move args to device
    args_on_device = tree_map(lambda x: to_device(x, device), workload.args)

    # 2. Move kwargs to device
    kwargs_on_device = tree_map(lambda x: to_device(x, device), workload.kwargs)

    # 3. Move model to device (if present)
    if workload.model is not None and hasattr(workload.model, "to"):
        workload.model = workload.model.to(device)
        if hasattr(workload.model, "tie_weights"):
            workload.model.tie_weights()

    # 4. Handle sharding — SKIPPED for CPU (device.type != "cpu" guard)

    # 5. Move compiled_executable to device
    workload.compiled_executable = to_device(workload.compiled_executable, device)

    # 6. Return NEW TorchWorkload with everything on-device
    return TorchWorkload(
        model=workload.model,
        executable=workload.executable,
        compiled_executable=workload.compiled_executable,
        args=args_on_device,
        kwargs=kwargs_on_device,
    )
```

**Sub-step details:**

| Sub-step | What It Does | For CPU |
|----------|--------------|---------|
| Move args | `tree_map` + `to_device` recursively walks all args and calls `.to(device)` on tensors | Usually a no-op (tensors are already on CPU) |
| Move kwargs | Same recursive walk for keyword arguments | Usually a no-op |
| Move model | Calls `model.to(device)` to move all parameters and buffers | Ensures weights are on CPU |
| Tie weights | Some HuggingFace models share weight tensors between layers. After `.to()`, aliases can break. `tie_weights()` re-establishes them | Prevents silent correctness bugs |
| Sharding | Marks tensors for multi-chip distribution via `xs.mark_sharding` | **Skipped entirely** — the `device.type != "cpu"` guard prevents it |
| Move compiled_executable | Calls `to_device()` on the `OptimizedModule` | If it has `.to()`, moves it; if it's a plain callable wrapper, returns unchanged |
| Return new workload | Creates a fresh `TorchWorkload` with all device-relocated pieces | Clean workload with everything on CPU |

**The `to_device` helper** (`torch_device_runner.py`):
- Handles **aliasing**: tracks `id(object)` → moved version, so if the same tensor appears twice, it's moved once and the same reference is reused.
- **Recurses up to depth 5**: handles nested structures (lists of dicts of tensors, etc.).
- Handles **custom objects**: if something has a `__dict__`, it recursively processes all attributes.
- Returns non-movable objects (ints, strings, etc.) unchanged.

### 4.4 Actual Execution

**File:** `tests/infra/runners/torch_device_runner.py`

```python
def _run_on_device(self, workload, device):
    with torch.set_grad_enabled(self.training_mode):
        result = workload.execute()
    return result
```

1. **Sets gradient context**: `torch.no_grad()` for inference (`training_mode=False`), `torch.enable_grad()` for training (`training_mode=True`).
2. **Calls `workload.execute()`** (from `workload.py`), which checks priority order:
   - `compiled_executable` → **used** (this is what `torch.compile` set)
   - `model` → fallback
   - `executable` → last resort

So it calls `compiled_executable(*args, **kwargs)`. **This is where lazy compilation fires on the first call.**

---

## 5. Phase 4 — Inductor Internals (First Run)

When `compiled_executable(*args, **kwargs)` is invoked for the first time, the full Dynamo + Inductor pipeline runs.

### 5.1 TorchDynamo — Bytecode Tracing

**Goal:** Convert Python model code into a clean graph of tensor operations.

**Step 1 — Intercept the call.**
The `OptimizedModule` wrapper's `__call__` hands control to TorchDynamo instead of running the original Python code directly.

**Step 2 — Bytecode analysis.**
Dynamo operates at the Python bytecode level (`.pyc` instructions). It uses a custom `sys.settrace`-based frame evaluator that intercepts every Python frame (function call) inside the model. It reads `LOAD_ATTR`, `CALL_FUNCTION`, `BINARY_ADD`, etc. instructions one by one.

**Step 3 — Symbolic tracing.**
Instead of actually executing tensor operations, Dynamo creates **symbolic proxies** (fake tensors that track shape/dtype but hold no data). When the code does `x = a + b`, Dynamo records a node `add(a_proxy, b_proxy)` in an **FX Graph** without computing anything.

```
Python code:       x = torch.relu(input @ weight + bias)

Dynamo records:    FX Node: mm(input, weight)
                   FX Node: add(mm_result, bias)
                   FX Node: relu(add_result)
```

**Step 4 — Handle Python control flow (graph breaks).**
If Dynamo encounters something it can't trace symbolically (data-dependent `if` statements, unsupported builtins, print statements with tensor values), it creates a **graph break** — splitting the model into multiple sub-graphs compiled independently with regular Python running in between.

For a clean `nn.Module.forward()`, there are typically **zero graph breaks**.

**Step 5 — Create guards.**
Dynamo records **guards** — conditions that must remain true for the compiled code to be reusable:
- Input tensor shapes (e.g., `args[0].shape == (1, 3, 224, 224)`)
- Input tensor dtypes (e.g., `args[0].dtype == torch.float32`)
- Input tensor device (e.g., `args[0].device == cpu`)
- Python variable values that affected control flow
- Model object type hasn't changed

**Output:** An `FX GraphModule` (clean, Python-free representation of tensor operations) + a set of guards.

### 5.2 Inductor Frontend — Graph Optimizations

**Goal:** Optimize the FX graph before generating code.

**Step 6 — Lower to Inductor IR.**
The FX graph's high-level PyTorch ops (`torch.mm`, `torch.relu`) are lowered into Inductor's own intermediate representation with more primitive operations closer to hardware instructions.

**Step 7 — Operator fusion.**
Inductor analyzes data dependencies and **fuses** operations that can run together:

```
Before fusion:              After fusion:
┌─────────┐                 ┌──────────────────┐
│  matmul  │                │                  │
└────┬─────┘                │  matmul + add    │
     │                      │  + relu          │
┌────▼─────┐                │  (single kernel) │
│   add    │                │                  │
└────┬─────┘                └──────────────────┘
     │
┌────▼─────┐
│   relu   │
└──────────┘

3 memory round-trips  →  1 memory round-trip
```

Without fusion, each op reads from memory, computes, writes back. With fusion, intermediate results stay in CPU registers/cache. For memory-bound ops, this yields significant speedups.

**Step 8 — Additional optimizations:**
- **Constant folding**: Pre-compute anything depending only on constants/weights.
- **Dead code elimination**: Remove ops whose outputs are never used.
- **Layout optimization**: Choose row-major vs column-major per tensor to minimize cache misses.
- **Common subexpression elimination**: If the same computation appears twice, compute it once.

### 5.3 Inductor Backend — CPU Code Generation

**Goal:** Turn the optimized IR into actual machine code.

**Step 9 — Generate C++/OpenMP source code.**
For CPU, Inductor generates C++ source code as a string. Key CPU-specific features in the generated code:

- `#pragma omp parallel for` — multi-threaded execution across CPU cores.
- `#pragma omp simd` — vectorized SIMD instructions (AVX2/AVX-512) processing 8–16 floats per instruction.
- **Tiling** — loops are broken into tile sizes that fit in L1/L2 cache.
- For matmuls, Inductor often calls optimized BLAS libraries (MKL, OpenBLAS) instead of generating raw loops.

**Step 10 — JIT compile the C++ code.**
The generated C++ source is written to a temp file and compiled:

```bash
g++ -O3 -march=native -fopenmp -shared -fPIC generated_kernel.cpp -o generated_kernel.so
```

| Flag | Purpose |
|------|---------|
| `-O3` | Maximum compiler optimizations |
| `-march=native` | Use the best instructions available on the specific CPU |
| `-fopenmp` | Enable OpenMP threading |
| `-shared -fPIC` | Produce a dynamically loadable shared library |

**Step 11 — Load the compiled library.**
Python loads the `.so` file via `ctypes` / `torch.utils.cpp_extension`, making the compiled C++ functions callable from Python.

### 5.4 Compiled Code Execution

**Step 12 — Run the compiled code.**
The compiled native functions are called with actual tensor data pointers. The CPU executes the optimized, fused, vectorized, multi-threaded machine code.

**Step 13 — Cache everything.**
Dynamo stores a cache entry:

```
Cache Entry = {
    guards:        [shape checks, dtype checks, device checks, ...],
    compiled_code: <pointer to loaded .so functions>,
    fx_graph:      <the graph, for debugging>
}
```

This cache lives on the `OptimizedModule` object.

---

## 6. Phase 5 — Cached Execution (Subsequent Runs)

### 6.1 Guard Checking

On every call after the first:

```
compiled_executable(*args, **kwargs)
  │
  ├── Step 1: Check guards (microseconds)
  │     ├── args[0].shape == (1, 3, 224, 224)?   ✓
  │     ├── args[0].dtype == float32?              ✓
  │     ├── args[0].device == cpu?                 ✓
  │     └── ... all guards pass?                   ✓
  │
  ├── Step 2: Cache HIT → skip Dynamo, Inductor, C++ compilation
  │
  └── Step 3: Call cached .so functions directly → return result
```

Guard checks are simple Python attribute comparisons taking **microseconds**.

### 6.2 Recompilation Triggers

| Scenario | What Happens |
|----------|--------------|
| Same shape, same dtype | Guards pass → cached code runs (fast) |
| Different shape (e.g., batch size changes) | Guards fail → full recompilation for new shape |
| Different dtype (e.g., float32 → bfloat16) | Guards fail → full recompilation |
| Same shape after recompile | Two cache entries exist — one per shape — both fast |
| Too many recompilations (default limit: 8) | Dynamo gives up, falls back to eager Python, logs a warning |

### Typical Timing Breakdown

| Phase | First Run | Subsequent Runs |
|-------|-----------|-----------------|
| Dynamo tracing | ~100–500ms | skipped |
| Inductor optimization | ~200–1000ms | skipped |
| C++ code generation | ~50–200ms | skipped |
| C++ JIT compilation (`g++ -O3`) | ~1–5 seconds | skipped |
| Loading .so | ~10–50ms | skipped |
| Guard checking | N/A | ~1–10μs |
| Actual execution | 10–500ms | 10–500ms |
| **Total** | **~2–7 seconds** | **~10–500ms** |

---

## 7. Result Return Path

After execution, the result propagates back up the call chain:

```
workload.execute()                                   → torch.Tensor
  ↑
TorchDeviceRunner._run_on_device(workload, device)   → torch.Tensor
  ↑
DeviceRunner.run_on_device(workload, CPU)             → torch.Tensor
  ↑
DeviceRunner.run_on_cpu(workload)                     → torch.Tensor
  ↑
ModelTester._run_on_cpu(workload)                     → torch.Tensor
  ↑
TorchModelTester._run_on_cpu(workload)                → torch.Tensor
  ↑  (JAX accelerator unmasked here on context exit)
  ↑
ModelTester._test_inference()  stores as `cpu_res`    → torch.Tensor
```

The returned `torch.Tensor` is a standard CPU tensor containing the model's forward pass output. It becomes the **golden reference** for TT device comparison.

---

## 8. How CPU Results Are Used

### Inference Flow

```python
# In ModelTester._test_inference():
cpu_res = self._run_on_cpu(self._workload)         # Golden reference
tt_res  = self._run_on_tt_device(self._workload)   # TT device result
result  = self._compare(tt_res, cpu_res)            # Compare within tolerance
```

### Training Flow

```python
# In TorchModelTester._test_training():
# Forward pass
cpu_res = self._run_on_cpu(self._workload)           # CPU forward golden
tt_res  = self._run_on_tt_device(self._workload)     # TT forward result

# Backward pass
self._run_on_cpu(cpu_backward_workload)               # CPU backward (populates grads)
self._run_on_tt_device(tt_backward_workload)           # TT backward

# Compare both
forward_result  = self._compare(tt_res, cpu_res)
backward_result = self._compare(tt_grads, cpu_grads)
```

---

## 9. Complete Call Chain Diagram

```
test()
 └── _test_inference()
      │
      ├── _compile_for_cpu(workload)
      │    └── compile_torch_workload_for_cpu(workload)
      │         ├── Pick model or executable
      │         └── workload.compiled_executable = torch.compile(..., backend="inductor")
      │              └── Returns OptimizedModule (lazy — NO compilation yet)
      │
      └── _run_on_cpu(workload)
           │
           ├── _mask_jax_accelerator()
           │    └── Monkey-patch torch.accelerator.is_available to hide JAX
           │
           └── ModelTester._run_on_cpu(workload)
                │
                └── DeviceRunner.run_on_cpu(workload)
                     │
                     └── DeviceRunner.run_on_device(workload, CPU)
                          │
                          ├── 1. connect_device(CPU)
                          │     └── TorchDeviceConnector → torch.device("cpu")
                          │
                          ├── 2. _put_on_device(workload, cpu)
                          │     └── _safely_put_workload_on_device()
                          │           ├── tree_map args → .to(cpu)
                          │           ├── tree_map kwargs → .to(cpu)
                          │           ├── model.to(cpu) + tie_weights()
                          │           ├── [sharding skipped for CPU]
                          │           ├── compiled_executable → to_device(cpu)
                          │           └── return NEW TorchWorkload(all on CPU)
                          │
                          └── 3. _run_on_device(device_workload, cpu)
                                │
                                ├── torch.set_grad_enabled(training_mode)
                                │
                                └── workload.execute()
                                     └── compiled_executable(*args, **kwargs)
                                          │
                                          ├── [1st call] FULL COMPILATION:
                                          │    Dynamo traces bytecode
                                          │      → FX Graph (symbolic tensor ops)
                                          │      → Guards recorded (shapes, dtypes, device)
                                          │      → Inductor optimizes (fusion, layout, CSE)
                                          │      → C++ source code generated
                                          │      → g++ -O3 JIT compile → .so binary
                                          │      → Load .so → execute → cache result
                                          │
                                          └── [Nth call] CACHED:
                                               Check guards → cache HIT → call .so → result
           │
           └── _unmask_jax_accelerator()
                └── Restore original torch.accelerator.is_available
                │
                └── return torch.Tensor (CPU golden result)
```

---

## 10. Key Source Files

| File | Role |
|------|------|
| `tests/infra/testers/base_tester.py` | Base tester: creates DeviceRunner, Evaluator |
| `tests/infra/testers/single_chip/model/model_tester.py` | Orchestrates init, compile, run, compare |
| `tests/infra/testers/single_chip/model/torch_model_tester.py` | Torch-specific: JAX masking, compile/run overrides, training flow |
| `tests/infra/utilities/utils.py` | `compile_torch_workload_for_cpu` — the `torch.compile` call |
| `tests/infra/runners/device_runner.py` | Abstract `run_on_cpu` → `run_on_device` orchestration |
| `tests/infra/runners/torch_device_runner.py` | `_safely_put_workload_on_device`, `to_device`, `_run_on_device` |
| `tests/infra/connectors/torch_device_connector.py` | `connect_device(CPU)` → `torch.device("cpu")` |
| `tests/infra/workloads/workload.py` | `Workload.execute()` — dispatches to `compiled_executable` |
| `tests/infra/workloads/torch_workload.py` | `TorchWorkload` — Torch-specific workload subclass |
| `tests/infra/runners/utils.py` | `@run_on_cpu` decorator for standalone functions |
| `tests/infra/runners/device_runner_factory.py` | Factory: framework → DeviceRunner |
| `tests/infra/connectors/device_connector_factory.py` | Factory: framework → DeviceConnector |
