# TT-XLA: Python-Level Host Memory Leak After TP LLM Tests

**Date**: 2026-03-01
**Repo**: tt-xla
**Branch**: kmabee/llms_in_galaxy_ci.testing
**Fix commit**: 9eb04f3fa (tests/conftest.py), 6e4b1801b (C++/test stability)

---

## Problem

Sequential tensor-parallel (TP) LLM tests accumulate ~26 GB of host RAM per large model
test in the test process. The memory is NOT released between tests, causing OOM
on machines with limited RAM when running multiple TP models sequentially (e.g. Qwen3-8B
followed by Qwen3-14B).

**None of these helped:**
- `gc.collect()` — Python garbage collector
- `libc.malloc_trim(0)` — glibc heap compaction
- `torch._dynamo.reset()` — clears dynamo compilation caches/XLAExecutor objects
- `xr.clear_computation_cache()` — TorchXLA computation cache
- Multiple `gc.collect()` passes

**Confirmed not the cause:**
- `gc.garbage` was empty (no uncollectable objects)
- No `__del__` methods on relevant classes (GraphModule, Graph, Node, etc.)
- `XLAExecutor` objects: 0 after `torch._dynamo.reset()` (executors ARE freed)

---

## Investigation Steps

### Step 1: smaps analysis

Used `/proc/self/smaps` and `smaps_rollup` to identify where 26 GB lives:

```
rss=  13785 MB  virt=  22131 MB  rw-p  [heap]    ← glibc heap
rss=   9504 MB  virt=   9516 MB  rw-p  (anon)    ← mmap (PJRT buffers)
rss=   1187 MB  virt=   1199 MB  rw-p  (anon)    ← one model weight shard
```

The heap holds ~13.8 GB, anonymous mmaps hold ~10.7 GB. Total ~24.5 GB out of 25.8 GB RSS.

### Step 2: XLA tensor scan

Added diagnostic to scan all non-CPU tensors via `gc.get_objects()`:

```
Live non-CPU tensors (xla/etc): total=93090.5 MB, large(>10MB) count=1091
  1187.0 MB: device=xla shape=(151936, 4096), dtype=torch.bfloat16  [embed table]
  1187.0 MB: device=xla shape=(151936, 4096), dtype=torch.bfloat16  [lm_head]
  ...  (1091 tensors total — full set of model weights × TP shards)
```

93 GB of *logical* XLA tensors alive (logical = device-side data), but only 26 GB of
host RSS. In TP mode, PJRT copies weight tensors to host for distribution — that's the 26 GB.

### Step 3: referrer tracing

Used `gc.get_referrers()` to find what holds these XLA tensors:

```
Referrers of XLA tensor shape=(151936, 4096):
  ← dict: {'val': FakeTensor(..., device='xla:0', ...)}  ← node.meta dict
    ← Node: p_model_embed_tokens_weight                  ← FX node
```

Tensors are in `node.meta['val']` of torch.fx.GraphModule nodes.

### Step 4: GraphModule scan

Scanned for live `torch.fx.GraphModule` objects:

```
Live torch.fx.GraphModule objects: 2
  GraphModule type=GraphModule    ← partitioned_graph (outer)
  GraphModule type=fused_0        ← fused XLA submodule
Functions with closures holding live GraphModules: 1
  function='extract_internal.<locals>.optimized_mod'
    defined at dynamo_bridge.py:544
    ← tuple: (<function optimized_mod>, <cell: fused_0>, fused_0())
    ← dict: {'inf': inf, 'nan': nan, ..., 'optimized_mod': <fn>}  ← forward.__globals__
    ← Node: optimized_mod                                           ← FX node target
```

**Key finding**: an `optimized_mod` closure (from `torch_xla/_dynamo/dynamo_bridge.py`)
captures `fused_0` (a `torch.fx.GraphModule` with XLA tensors in `node.meta['val']`).

### Step 5: root cause in dynamo_bridge.py

Reading `partition_fx_graph_for_cpu_fallback` (lines 798–808):

```python
for node in partitioned_graph.graph.nodes:
    if node.op == "call_module" and "fused_" in node.name:
        fused_module = getattr(partitioned_graph, node.name)
        partitioned_graph.delete_submodule(node.target)
        with partitioned_graph.graph.inserting_after(node):
            new_node = partitioned_graph.graph.call_function(
                extract_internal(fused_module),   # ← stores closure as node TARGET
                node.args, None)
            node.replace_all_uses_with(new_node)
        partitioned_graph.graph.erase_node(node)
partitioned_graph.recompile()
return partitioned_graph
```

`extract_internal(fused_module)` returns `optimized_mod`, a closure that:
1. Is stored as the **target** of a `call_function` FX node in `partitioned_graph`
2. After `recompile()`, is also in `GraphModuleImpl.forward.__globals__['optimized_mod']`

Reading `extract_internal` (line 513) and `extract_graph_helper` (line 339):

```python
def extract_internal(xla_model):
    sym_constants_to_graph_vars = {}  # ← captured by closure!
    (... args_and_out ...) = extract_graph_helper(xla_model, sym_constants_to_graph_vars)

    def optimized_mod(*args):
        nonlocal xla_model
        nonlocal sym_constants_to_graph_vars  # ← holds GraphInputMatcher!
        ...
    return optimized_mod
```

In `extract_graph_helper`:

```python
(graph_input_tensor_ids,
 graph_input_xla_values,        # ← ALL model-weight XLA tensors!
) = torch_xla._XLAC._get_tensors_xla_device_data_node(args_and_out_tensor_only)

graph_input_matcher = GraphInputMatcher(
    tensor_id_to_arg_idx,
    graph_input_tensor_ids,
    graph_input_xla_values,     # ← stored here
    xla_args_tensor_ids)

vars_to_return = (..., graph_input_matcher, ...)
sym_constants_to_graph_vars[sym_constants] = vars_to_return  # ← cached!
```

`GraphInputMatcher.graph_input_xla_values` stores **all model weights as XLA tensors**
(the "constant" graph inputs that don't change between calls). For 8B Qwen3 TP,
this is 1091 tensors totalling ~26 GB.

### Step 6: complete reference chain

```
optimized_mod closure
  ├── sym_constants_to_graph_vars (dict)
  │   └── vars_to_return (tuple)
  │       └── graph_input_matcher (GraphInputMatcher)
  │           └── graph_input_xla_values (list)
  │               └── [XLA tensor 1 (1187 MB), XLA tensor 2, ..., ×1091] ← 26 GB
  └── xla_model = fused_0 (GraphModule)
      └── graph.nodes[i].meta['val'] = XLA tensors  (secondary path)

partitioned_graph (GraphModule)
  └── graph.nodes
      └── Node(op='call_function', target=optimized_mod)  ← owns closure ref
      AND: forward.__globals__['optimized_mod'] = optimized_mod  ← after recompile()
```

### Step 7: why gc.collect() fails

The reference cycle is:
- `partitioned_graph` ↔ `GraphModuleImpl` (per-instance class, via `type()` and `_wrapped_call`)
- `GraphModuleImpl.forward.__globals__['optimized_mod']` → `optimized_mod`

Python's cyclic GC COULD collect this cycle, but something external holds it alive.
The external holder is never definitively identified, but clearing `sym_constants_to_graph_vars`
breaks the chain and frees the 26 GB regardless.

---

## Attempted Fixes (that didn't work)

### Attempt 1: clear node.meta on all GraphModules

```python
for obj in gc.get_objects():
    if isinstance(obj, torch.fx.GraphModule):
        for node in obj.graph.nodes:
            node.meta.clear()
gc.collect()
```

**Result**: cleared 18,786 nodes across 5 GraphModules. But RSS stayed at 26,824 MB.

**Why it didn't work**: The XLA tensors are NOT primarily in `node.meta['val']`
(which holds `FakeTensor` wrappers). The *actual* 26 GB is in
`GraphInputMatcher.graph_input_xla_values`. Clearing node.meta releases secondary
references only; `sym_constants_to_graph_vars` still holds the primary references.

---

## The Fix

In `tests/conftest.py`, a `_release_dynamo_bridge_tensors()` function is called from
`run_around_tests` fixture teardown, after `torch._dynamo.reset()`:

```python
def _release_dynamo_bridge_tensors():
    """Workaround for torch_xla leak: after torch._dynamo.reset(), GraphInputMatcher
    objects and their parent caches survive, holding all model-weight XLA tensors
    (~26 GB for 8B TP). We find these by type and clear their parent dicts.
    """
    from torch_xla._dynamo.dynamo_bridge import GraphInputMatcher

    for obj in gc.get_objects():
        if isinstance(obj, torch.fx.GraphModule):
            if getattr(obj, "xla_args", None) is not None:
                obj.xla_args = None
        if isinstance(obj, GraphInputMatcher):
            for ref in gc.get_referrers(obj):
                if isinstance(ref, tuple):
                    for d in gc.get_referrers(ref):
                        if isinstance(d, dict):
                            d.clear()
```

**How it works**:
- Imports `GraphInputMatcher` class from torch_xla for type-safe matching (no string matching)
- Scans all Python objects for live `GraphInputMatcher` instances
- Traces referrers: `GraphInputMatcher` ← `tuple` ← `dict` (the `sym_constants_to_graph_vars` cache)
- Clears each parent dict, releasing all cached GraphInputMatcher objects and their XLA tensors
- Also clears `xla_args` on surviving `GraphModule` objects (cached input tensors that pin model weights)
- `gc.collect()` in the existing `memory_usage_tracker` fixture handles cycle collection

**Results**:

| Metric | Before fix | After fix |
|--------|-----------|-----------|
| RSS after 8B TP test | 26,824 MB | 1,758 MB |
| RSS after 0.6B TP test | 2,794 MB | 1,627 MB |
| Heap (smaps) after 8B | 13,785 MB | 484 MB |
| 9.5 GB anon mmap | present | gone |

---

## Key Files

- `tests/conftest.py` — `run_around_tests` fixture (the fix)
- `venv/lib/.../torch_xla/_dynamo/dynamo_bridge.py` — `extract_internal()`, `extract_graph_helper()`, `partition_fx_graph_for_cpu_fallback()`
- `venv/lib/.../torch_xla/_dynamo/dynamo_bridge.py:35` — `GraphInputMatcher` class

---

## Diagnostic Infrastructure Added

The `memory_usage_tracker` fixture in `tests/conftest.py` now writes to
`/tmp/mem_diag_<pid>.log` when `--log-memory` is passed:

- Live CPU tensor sizes (total + large tensors)
- Live XLA/non-CPU tensor sizes (total + large tensors with device/shape/dtype)
- Referrer tracing for top-3 largest XLA tensors (2 levels deep)
- Live `torch.fx.GraphModule` count + closures holding them
- Live `XLAExecutor` count + referrers
- Live `torch.nn.Module` count + param counts
- `smaps_rollup` breakdown (Rss, Private_Dirty, Anonymous, etc.)
- Top-15 smaps regions by RSS

This is only active when `--log-memory` is explicitly passed; zero overhead otherwise.

---

## Notes

- This is distinct from the C++ program cache leak (separate fix in
  `loaded_executable_instance.cc`) — that leak affects the WORKER process (distributed
  runtime), while this leak affects the TEST PROCESS.
- The fix uses `isinstance(obj, GraphInputMatcher)` for type-safe matching. If torch_xla
  renames or removes this class, the import will fail loudly rather than silently no-op.
- The `--forked` mode (`pytest-forked`) would also avoid this leak since each test
  runs in a subprocess, but has its own overhead and failure modes.

---

## Related Fixes Found During the Same Investigation

These four issues were found and fixed in the same investigation pass (commit `6e4b1801b`).
They are independent bugs but all surfaced through the same TP LLM test runs.

### 1. C++ Program Cache Never Freed for Distributed Runtime

**File**: `pjrt_implementation/src/api/loaded_executable_instance.cc`

**Problem**: `LoadedExecutableInstance::~LoadedExecutableInstance()` guarded
`clearProgramCache()` with:

```cpp
if (device && getCurrentHostRuntime() == HostRuntime::Local &&
    isProgramCacheEnabled(*device)) {
```

For tensor-parallel (TP) models the runtime is distributed (`HostRuntime::Distributed`),
not local. The `HostRuntime::Local` guard made the cache never cleared in the WORKER
PROCESS across TP tests. This is a separate leak from the Python-level one — it affects
the worker processes, not the test process.

**Fix**: Remove the `getCurrentHostRuntime() == HostRuntime::Local &&` guard:

```cpp
if (device && isProgramCacheEnabled(*device)) {
```

**Note**: This leak was discovered via `smaps` analysis of the worker process (separate
from the test process analysis). The C++ program cache stores compiled flatbuffer binaries
in device memory; without clearing it, memory usage in the worker process grows with each
new model compiled.

---

### 2. MLIR Context Accumulates BumpPtrAllocator Memory Across Compilations

**Files**: `pjrt_implementation/src/api/module_builder/module_builder.cc`,
`pjrt_implementation/inc/api/module_builder/module_builder.h`,
`pjrt_implementation/src/api/client_instance.cc`

**Problem**: `ModuleBuilder` holds a single `mlir::MLIRContext` that lives for the entire
lifetime of the client. Each call to `buildModule()` runs the full VHLO→SHLO→TTIR→TTNN
compilation pipeline through MLIR pass managers. MLIR's internal `BumpPtrAllocator` (used
for IR storage) cannot free individual allocations — memory is only released when the
context is destroyed. Over multiple model compilations, this accumulates.

**Fix**: Added `ModuleBuilder::resetContext()`, which destroys and recreates the
`MLIRContext`, releasing all bump-allocated IR:

```cpp
void ModuleBuilder::resetContext() {
  m_context = std::make_unique<mlir::MLIRContext>();
  registerDialectsInContext(*m_context);
}
```

Called from `ClientInstance::compileMlirProgram()` immediately after `buildModule()` returns
(after all `OwningOpRef` locals are destroyed, which is important — `OwningOpRef` destructs
release IR into the context's allocator, and the context must outlive them):

```cpp
auto [status, image] = m_module_builder->buildModule(...);
m_module_builder->resetContext();  // free BumpPtrAllocator
```

Also extracted `registerDialectsInContext(mlir::MLIRContext&)` as a static helper so
dialect registration is consistent between the constructor and `resetContext()`.

**Note**: Size logging was also added (`compileMlirProgram: mlir_code size = X.Y MB`) to
make it easy to see how large each compilation's input is.

---

### 3. SIGABRT After Test Failure from XLA Runtime Re-entry

**Files**: `tests/runner/test_models.py`, `tests/runner/test_utils.py`

**Problem**: `test_all_models_torch` calls `get_xla_device_arch()` inside a `finally`
block for performance benchmarking setup. If the test had already failed and the device
was in a bad state (closed or partially torn down), calling
`xr.global_runtime_device_attributes()` inside `get_xla_device_arch()` caused a SIGABRT
in the worker process — which killed the entire test runner.

Two separate sub-problems:

**3a. Perf benchmarking runs even on failure**
```python
# Before:
if framework == Framework.TORCH and run_mode == RunMode.INFERENCE:
    # runs on both pass and fail paths

# After:
if framework == Framework.TORCH and run_mode == RunMode.INFERENCE and succeeded:
    # only runs when the test actually passed
```

**3b. `get_xla_device_arch()` called repeatedly, re-entering XLA runtime each time**

`get_xla_device_arch()` in `test_utils.py` called `xr.global_runtime_device_attributes()`
on every invocation. Added module-level caching so XLA is only queried once:

```python
_cached_device_arch: Optional[str] = None

def get_xla_device_arch():
    global _cached_device_arch
    if _cached_device_arch is not None:
        return _cached_device_arch
    # ... existing arch detection logic ...
    _cached_device_arch = arch
    return _cached_device_arch
```

This also eliminates redundant runtime queries across multiple tests in the same session.

---

### 4. Explicit Model Object Cleanup After Each Test

**File**: `tests/runner/test_models.py`

**Problem**: After each test, the `tester` object (and its `_workload`) held references to
the model, compiled executable, and input tensors. Python reference counting wouldn't free
these until the tester object itself was GC'd, which might not happen promptly — especially
if the reference cycle around the GraphModule (described in the main fix above) prevented
timely cleanup.

**Fix**: Explicit null-out in the `finally` block:

```python
if tester is not None:
    if getattr(tester, "_workload", None) is not None:
        tester._workload.model = None
        tester._workload.compiled_executable = None
        tester._workload.args = []
        tester._workload.kwargs = {}
        tester._workload.shard_spec_fn = None
    tester._model = None
    tester._input_activations = None
```

This breaks the reference chain from the tester → model weights promptly after each test,
complementing the `sym_constants_to_graph_vars` cleanup in conftest.py.
