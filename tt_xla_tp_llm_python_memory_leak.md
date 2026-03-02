# TT-XLA: Python-Level Host Memory Leak After TP LLM Tests

**Date**: 2026-03-01
**Repo**: tt-xla
**PR**: #3521

---

## Problem

Sequential tensor-parallel (TP) LLM tests accumulate host RAM proportional to model size
per test in the test process. The memory is NOT released between tests, causing OOM
when running multiple TP models sequentially (e.g. ~258 GB retained after GPT-OSS-120B
on Galaxy, ~26 GB after Qwen3-8B on N300).

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

### Attempt 2: clear only GraphInputMatcher.graph_input_xla_values

```python
from torch_xla._dynamo.dynamo_bridge import GraphInputMatcher
for obj in gc.get_objects():
    if isinstance(obj, GraphInputMatcher) and obj.graph_input_xla_values:
        obj.graph_input_xla_values.clear()
```

**Result**: RSS stayed at 26,824 MB.

**Why it didn't work**: Clearing the tensor list inside each matcher is not enough.
The parent `sym_constants_to_graph_vars` dict must be cleared entirely — other entries
in the cached tuple (e.g. `args_and_out`) also hold tensor references. Additionally,
`xla_model.xla_args` on the captured GraphModule pins model weights independently.

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

Galaxy (32-chip) — GPT-OSS-120B followed by Llama-3.1-70B:
- Process RSS after GPT-OSS-120B cleanup: 257,829 MB → 30,625 MB
- System free at Llama-70B start: 204,532 MB → 432,060 MB

N300 (2-chip) — Qwen3-8B TP:
- Process RSS after cleanup: ~27,000 MB → ~1,600 MB

---

## Key Files

- `tests/conftest.py` — `run_around_tests` fixture (the fix)
- `venv/lib/.../torch_xla/_dynamo/dynamo_bridge.py` — `extract_internal()`, `extract_graph_helper()`, `partition_fx_graph_for_cpu_fallback()`
- `venv/lib/.../torch_xla/_dynamo/dynamo_bridge.py:35` — `GraphInputMatcher` class

---

## Diagnostic Changes

The `memory_usage_tracker` fixture in `tests/conftest.py` now includes process RSS
in its `--log-memory` output at both sampling positions (after test, after gc).

---

## Notes

- This is a workaround, not a proper fix. The proper fix is for torch_xla's `openxla`
  backend to implement a `reset()` method so `torch._dynamo.reset()` can notify it to
  clean up. Dynamo already calls `backend.reset()` via `_reset_guarded_backend_cache()`,
  but the `openxla` backend is a bare `aot_autograd` partial with no `reset()` method.
- The fix uses `isinstance(obj, GraphInputMatcher)` for type-safe matching. If torch_xla
  renames or removes this class, the import will fail loudly rather than silently no-op.
- There is also a separate C++ program cache leak in the worker process (distributed
  runtime) — `LoadedExecutableInstance::~LoadedExecutableInstance()` guards
  `clearProgramCache` with `HostRuntime::Local`, so the cache is never cleared for TP.
  This is tracked separately.
- The `--forked` mode (`pytest-forked`) would also avoid this leak since each test
  runs in a subprocess, but has its own overhead and failure modes.
