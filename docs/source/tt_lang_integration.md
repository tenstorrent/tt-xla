# tt-lang integration in tt-xla

This document describes how user-authored
[tt-lang](https://github.com/tenstorrent/tt-lang) kernels get from PyTorch
source code down to running on Tenstorrent hardware through tt-xla.

The Python surface lives in a single file: `python_package/tt_torch/tt_lang.py`.
The torch custom op it emits is registered in
`python_package/tt_torch/custom_ops.py` alongside the rest of `torch.ops.tt.*`.

## Author-facing API

```python
import tt_torch
import ttl  # tt-lang

@tt_torch.kernel(kernel_id="moe.routed_mlp.v1", arg_roles=("in", "in", "out"))
@ttl.operation(grid=(8, 8))
def routed_mlp(lhs, rhs, out):
    ...

# On XLA tensors:
routed_mlp(lhs, rhs, out)
```

The decorator does three things at call time:

1. Validates inputs and copies argument-role metadata into a process-global
   registry keyed by `kernel_id`.
2. Emits `stablehlo.custom_call @tt.tt_lang_op` whose `frontend_attributes`
   carry the metadata the plugin will need at compile/execute time.
3. Copies the functional custom-call results back into the user's
   pre-allocated `"out"` operands so the mutation-style API is honored.

The op is always kept inside the XLA graph; there is no host escape hatch.
Shardy, TTIR legalization, and TTNN lowering all see it as a normal op.

## Pipeline

```text
PyTorch model
   |  @tt_torch.kernel(kernel_id=...)  -- registers callable, emits custom op
   v
stablehlo.custom_call @tt.tt_lang_op
   { kernel_id, arg_roles, version_tag, shard_spec }
   |
   v  pjrt_plugin_tt::ModuleBuilder
SHLO / Shardy frontend          (custom call survives untouched)
   |
   v
StableHLO -> TTIR               (ttir.tt_lang_op, attributes preserved)
   |
   v
TTIR -> TTNN                    (ttnn.tt_lang_op, kernel_artifact = empty)
   |
   v  ModuleBuilder::resolveTtLangKernels (compile time, post-TTNN)
plugin walks the module for `ttnn.tt_lang_op`s and calls
    tt_torch.tt_lang.resolve_kernel(kernel_id, version_tag, shapes, dtypes, ...)
through the embedded Python interpreter (host's libpython, GIL acquired).
   |
   v
resolved bytes attached as the `kernel_artifact` attribute on the op
   |
   v
TTNN flatbuffer emitter embeds the artifact into the executable
   |
   v
PJRT executable returned to torch-xla; runtime launches with the
already-bound kernel -- no Python on the hot path.
```

## What ships in tt-xla today

| Piece | Location | Status |
|---|---|---|
| `torch.ops.tt.tt_lang_op` (variadic custom op + fake + autograd-raises) | `python_package/tt_torch/custom_ops.py` | done |
| `@tt_torch.kernel` decorator | `python_package/tt_torch/tt_lang.py` | done |
| `kernel_id -> KernelEntry` registry | `python_package/tt_torch/tt_lang.py` | done |
| `resolve_kernel(...)` entry point | `python_package/tt_torch/tt_lang.py` | **stub** -- raises `NotImplementedError`; signature is stable for the plugin to call against |
| Embedded-Python bridge (`pybind11::embed`, calls `tt_torch.tt_lang.resolve_kernel`) | `pjrt_implementation/{inc,src}/api/module_builder/tt_lang_bridge.{h,cc}` | done |
| `ModuleBuilder::resolveTtLangKernels(...)` compile-time hook (thin delegator to the bridge) | `pjrt_implementation/src/api/module_builder/module_builder.cc` | done -- no-op until tt-mlir emits `ttnn.tt_lang_op` |

The bridge translation unit (`tt_lang_bridge.cc`) is the only place that
depends on pybind11 / libpython. It is compiled with `-frtti` (the rest of
`TTPJRTApi` keeps `-fno-rtti`, inherited from LLVM). `pybind11::embed`
propagates `libpython3.12.so.1.0` into the plugin's `DT_NEEDED`; at runtime
the host Python (JAX or torch.compile) has already loaded libpython, so the
linker entry resolves to the running interpreter and we just acquire the
GIL -- we never call `Py_Initialize`.

## StableHLO contract

`stablehlo.custom_call @tt.tt_lang_op` is what flows out of PyTorch/XLA.
Its operands are the input tensors followed by the pre-allocated outputs
(so their layouts/types are visible to Shardy). The op has one result per
`"out"`-tagged operand with matching shape and dtype.

`frontend_attributes` (all strings):

| attribute | meaning |
|---|---|
| `kernel_id` | Stable identifier the plugin will pass back to `resolve_kernel`. |
| `arg_roles` | Comma-separated per-operand role, e.g. `"in,in,out"`. |
| `version_tag` | Hash of kernel source; the plugin sends it back so the resolver can refuse stale matches. |
| `shard_spec` | Optional opaque sharding hint (empty string when unused). |

Invariants downstream tooling must preserve:

* All four attributes survive the SHLO / Shardy frontend untouched.
* Operand layout / dtype / shape can be **refined** by Shardy (that is the
  point of leaving the op opaque), but operand **count** and **order**
  must not change.
* Result count equals the number of `"out"` entries in `arg_roles`, in
  declaration order.

## What still needs to be done

The Python side is intentionally minimal until the cross-stack pieces
land. Each item below is independent; pick them up in the order that
matches available cycles.

### 1. tt-mlir: legalize the custom call (done in the standalone tt-mlir repo, awaiting submodule bump here)

* Recognize `stablehlo.custom_call @tt.tt_lang_op` and lower it to a new
  `ttir.tt_lang_op` carrying the `frontend_attributes` as op-local named
  attributes:
  * `kernel_id`
  * `version_tag`
  * `arg_roles`
  * `shard_spec`
* Lower `ttir.tt_lang_op` to `ttnn.tt_lang_op` with the same four
  attributes preserved plus an optional `kernel_artifact: DenseI8ArrayAttr`
  initially left empty; operand types reflect post-Shardy local shapes /
  dtypes / layouts.

`ttnn.tt_lang_op` stubs the `OpModelInterface` (returns
`NoNeedForConstraintAPI`) and `WorkaroundInterface` (returns the no-op
default) because the kernel internals are opaque to the TTNN cost model.

### 2. Plugin: runtime resolve hook (done)

`ModuleBuilder::resolveTtLangKernels` runs between TTIR -> TTNN conversion
and the runtime backend handoff. It is a thin delegator to
`tt_lang_bridge::resolveKernels`, which:

1. Walks the post-TTNN module for ops named `ttnn.tt_lang_op`.
2. Acquires the GIL on the host Python (the one that loaded the plugin)
   and imports `tt_torch.tt_lang.resolve_kernel` once per pass.
3. For each op, reads the four named attributes, builds per-operand
   `(shape, dtype, layout)` triples from the op's operand types, and
   calls `resolve_kernel(...)` through pybind11.
4. Attaches the returned `bytes` back onto the op as the
   `kernel_artifact` `DenseI8ArrayAttr`.

What this still needs:

* tt-mlir submodule bump in `third_party/tt-mlir` to pick up the
  `ttnn.tt_lang_op` definition (the walk uses the op-name string, so the
  plugin will keep compiling against older snapshots; it just won't find
  anything to resolve). **Done** -- pinned at the
  `jackzhang/tt_lang_integration` tip which contains the dialect ops, the
  StableHLO -> TTIR / TTIR -> TTNN conversions, and the flatbuffer
  emitter described below.
* Layout hand-off in `tt_lang_bridge.cc` (T6). **Done** -- the bridge
  now stringifies each operand's `ttnn.ttnn_layout` encoding and passes
  it through to `resolve_kernel`; the Python side records it under
  `operand_metadata.layouts` in the artifact JSON so the flatbuffer
  emitter can consume it later.

### 3. Plugin: Python callback mechanism (done -- embedded interpreter)

The bridge embeds Python directly via `pybind11::embed` rather than going
through a separate `dlopen`'d shim. Rationale:

* The plugin is always loaded into a running Python process (JAX /
  torch.compile invokes us through PJRT). A separate shim .so would just
  re-enter the same interpreter we are already inside.
* `pybind11::embed` propagates `libpython3.12.so.1.0` into the plugin's
  `DT_NEEDED`; at runtime that resolves to the host's already-loaded
  copy. We never call `Py_Initialize`.
* The bridge lives in a single `.cc` (`tt_lang_bridge.cc`) compiled with
  `-frtti` so pybind11's typeid usage works while the rest of `TTPJRTApi`
  keeps `-fno-rtti` (inherited from LLVM/MLIR).

If `tt_torch.tt_lang` isn't importable from the host Python (e.g. the
plugin was loaded by a non-tt_torch Python), compilation fails with a
descriptive error pointing at the missing wheel.

### 4. tt-lang: compile driver (done)

`resolve_kernel` drives tt-lang's existing compile path with mock torch
tensors and the `TTLANG_COMPILE_ONLY=1` env var, monkey-patching
`ttl.ttl_api._compile_kernel` to capture the resulting
`CompiledTTNNKernel`. The captured bundle is serialized into a JSON byte
blob (kernel C++ sources, thread types, configs/CB/core-ranges, tensor
indices, plus `operand_metadata` recording the shapes/dtypes/layouts
the kernel was compiled against). See
`python_package/tt_torch/tt_lang.py::_serialize_compiled_kernel` and the
versioned `_ARTIFACT_FORMAT_VERSION` constant; the tt-mlir flatbuffer
emitter (T4) decodes the same schema.

This intentionally reaches into tt-lang's private `_compile_kernel` to
avoid changing tt-lang for the POC. Once tt-lang exposes a stable
compile API (e.g. `ttl.bridge.compile_for_tt_xla`) we can drop the
monkey-patch and the env-var dance.

### 4b. TTNN flatbuffer emitter (T4)

`tt-mlir`'s `TTNNToFlatbuffer.cpp` contains
`createOp(FlatbufferObjectCache &, TtLangOp)`, which parses the
`kernel_artifact` JSON (gated on `format_version == 1`) and emits a
`GenericOp` flatbuffer record:

* one `KernelDescriptor` per `kernels[*]` entry, with
  * the kernel's `cpp_source` **bytes embedded directly** (`SourceType::SOURCE_CODE`),
    so the flatbuffer is self-contained — no `/tmp` paths, survives AOT
    reload and cross-machine deployment;
  * the matching `ComputeKernelConfig` / `ReaderKernelConfig` /
    `WriterKernelConfig` populated from `kernel_config` (math fidelity,
    `fp32_dest_acc_en`, `dst_full_sync_en`, etc. round-trip
    field-for-field);
  * `compile_time_args` = `[KernelArgCBBufferIndex(i) for i in 0..num_cbs-1]`
    for compute kernels; for NOC kernels we additionally append one
    `KernelArgTensorAccessorArgs` *marker* per operand of the op (in
    operand declaration order). Each marker carries the operand's
    index into the surrounding `GenericOp::io_tensors` array; the
    runtime expands it by calling
    `::tt::tt_metal::TensorAccessorArgs(io_tensors[i].buffer()).get_compile_time_args()`
    against the live buffer at launch time (see "Runtime-derived
    TensorAccessor args" below). No values are baked at MLIR-translate
    time.
  * `common_runtime_args = [KernelArgBufferAddressOfTensor(i) for i in tensor_indices]`,
    so the runtime resolves the actual buffer address from the
    surrounding `GenericOp::io_tensors` at launch time;
* one `KernelCBDescriptor` per `cb_configs[*]` entry, with
  `total_size`, `data_format` (mapped to `ttcore.DataType` enum), and
  `page_size` carried straight through from
  `_serialize_cb_config`. CB sizing mirrors
  `tt-lang/kernel_runner.py::build_cb_descriptors` exactly so the
  flatbuffer is byte-equivalent to what tt-lang would have built at
  native launch time;
* `core_ranges` constructed from the artifact's
  `{"start": [x, y], "end": [x, y]}` rectangle (currently a single
  rectangle; tt-lang only emits one).

#### Runtime-derived TensorAccessor args

tt-lang data-movement kernels are JIT-compiled with
`compile_time_args = [<CB indices>, <TensorAccessor args>...]`. The
TensorAccessor block normally comes from
`ttnn.TensorAccessorArgs(buffer).get_compile_time_args()`, which needs a
real device-side `Buffer`. The plugin process cannot construct one (see
"Device-less compile path" below), so we derive these compile-time args
**at runtime from the live buffer** the GenericOp executes against.

The mechanics, end to end:

1. **Schema** (`generic_op.fbs`):
   ```
   table KernelArgTensorAccessorArgs {
     operand_index: uint32;       // index into GenericOp.io_tensors
   }
   union KernelArgType { ..., KernelArgTensorAccessorArgs }
   ```
   The marker is the only `KernelArg*` variant that expands to a
   variable number of uint32s at launch (everything else is 1:1).
2. **Emitter** (`TTNNToFlatbuffer.cpp::createOp(TtLangOp)`): for each
   NOC kernel, append one marker per operand of the op in declaration
   order. The marker stores the io_tensors index, computed from the
   `inIndices`/`outIndices` walk so the runtime sees a stable mapping
   no matter how io_tensors is reordered (today: ins first, then
   outs). The artifact itself carries no TensorAccessor values.
3. **Runtime** (`runtime/lib/ttnn/operations/generic/generic_op.cpp`):
   `createKernelArgs` expands `KernelArgTensorAccessorArgs` by pulling
   `io_tensors[operand_index].buffer()` and calling
   `::tt::tt_metal::TensorAccessorArgs(buffer).get_compile_time_args()`,
   then `insert`ing the resulting `vector<uint32_t>` into the
   kernel's compile-time args. The buffer is the real, allocated
   `tt_metal::Buffer*` the program will launch against, so the
   compile-time args match the shard spec / bank coords / page size
   / alignment exactly.

This avoids three problems an offline derivation would have:

* No synthesizer table to keep in sync with
  `tt_metal/impl/buffers/tensor_accessor_args.cpp` (sharded layouts,
  packed-tile dtypes, blackhole alignments, etc. all "just work"
  because tt-metal computes them at runtime).
* No "open ttnn in the plugin process to compute args" race against
  PJRT's own command-queue init.
* No dead synthesized values inside the flatbuffer: artifacts are
  smaller, and re-deploying the same flatbuffer to a chip with a
  different bank topology does not require re-resolving the kernel.

**Known gaps still blocking on-silicon execution for non-trivial kernels:**

1. **Multi-rectangle CoreRangeSets.** tt-lang only emits a single
   rectangle today, so the artifact schema models `core_range` as one
   `{start, end}` pair. When tt-lang gains multi-rectangle kernels the
   schema bumps to `core_range_set: [{start, end}, ...]`.

2. **PipeNet semaphores.** `num_pipe_nets` is carried in the artifact
   but the emitter currently emits an empty semaphore list. Kernels
   that use `ttl.PipeNet` for cross-thread synchronisation need a
   future schema entry that lists each semaphore's id / core_range /
   initial_value, mirroring
   `kernel_runner.py::run_kernel_on_device`'s semaphore loop.

3. **Sharded memory_config parsing.** `_ttnn_memory_config_from_layout`
   currently only distinguishes DRAM vs L1 (both interleaved) -- the
   only two cases tt-lang's compile path accepts today. When tt-lang
   grows sharded-kernel support we need a full parser that threads
   grid + shard_spec through to `ttnn.MemoryConfig(...)`.

4. **Reader vs writer NOC distinction.** The emitter writes
   `ReaderKernelConfig` for the first noc kernel (NCRISC) and
   `WriterKernelConfig` for the second (BRISC), matching tt-lang's
   `_compile_ttnn_kernel` assignment. The metal runtime maps these to
   `RISCV_1/Noc1` and `RISCV_0/Noc0` respectively.

Simple value-blind tt-lang kernels (elementwise, reductions, matmul
without auto-padded TensorAccessor reads) run end-to-end on silicon
when invoked with DRAM / interleaved-L1 operands. (1)–(3) gate broader
coverage.

5. **Device-less compile path (DEMO HACK, currently shipped).** tt-lang's
   compile-only path doesn't actually need a live chip; it only needs
   `(shape, dtype, layout, memory_space, grid)` metadata. The reason
   the bridge used to call `ttnn.open_device(0)` was that tt-lang's
   compile entry point insists on real `ttnn.Tensor` arguments
   (strict `isinstance(arg, ttnn.Tensor)`, plus a
   memory_space-must-be-L1-or-DRAM check), and the only legitimate way
   to produce one is `ttnn.from_torch(..., device=<a real device>)`.

   Opening that ttnn device in the same process as the plugin caused
   a `TT_FATAL: binary not found` at `kernel.cpp:Kernel::binaries`:
   both ttnn and the PJRT plugin register their own dispatch firmware
   in their own `tt::tt_metal::Program` objects, and the second
   consumer of device 0 in the process can't find its kernel binaries
   in the first consumer's cache. (Aside: even arranging for the two
   to share a single `libtt_metal.so` via `SONAME` dedup -- using the
   vendored ttnn at `third_party/.../tt-metal/ttnn` -- doesn't help,
   because the conflict is over per-`Program` runtime state, not over
   which `.so` the symbols resolve through.)

   **Current shipping workaround:**
   `python_package/tt_torch/tt_lang.py` defines `_StubTtnnTensor` /
   `_StubTtnnDevice` / `_StubMemoryConfig` /  `_StubGridSize`
   duck-typed stand-ins, and `_drive_ttl_compile` monkey-patches
   `ttl.ttl_api.is_ttnn_tensor` so the stubs pass tt-lang's
   `isinstance` gate for the duration of the compile. The plugin
   process never imports ttnn; the PJRT path is the only consumer of
   device 0, and the conflict is gone. Grep the codebase for
   `DEMO HACK` to find every site.

   *NOC kernels and TensorAccessor compile-time args:* the TensorAccessor
   block is derived at runtime from the live buffer (see
   "Runtime-derived TensorAccessor args" above), so the device-less
   compile path does not need to compute these values. Sharded /
   row-major / packed-tile / cross-architecture operands all work
   transparently because the runtime sees the real layout.

   **Remaining work (still warrants the DEMO HACK label):** tt-lang
   should grow a `_compile_ttnn_kernel_from_spec(specs)` entry point
   that takes `(shape, dtype, layout, memory_space, arch)` tuples
   instead of real tensors. The existing `_compile_ttnn_kernel(args)`
   becomes a thin shim that derives specs from tensors and calls the
   new function. No API break. Patch is ~50 LOC in
   `tt-lang/python/ttl/ttl_api.py`; the validation paths
   (`_require_device`, `_detect_memory_space_from_tensor`,
   `_is_mesh_tensor`) gain spec-aware overloads. Once that's
   available, the bridge can replace its `is_ttnn_tensor`
   monkey-patch with a clean call into the new entry point.

6. **`ttnn.tt_lang_op` layout workaround.** The TTNN layout pipeline
   asks each op "what layout do you need for each operand?" through
   `getOperandsWorkarounds()`. The original `ttnn.tt_lang_op`
   declaration returned `createEmptyTTNNOperandsWorkarounds()`, which
   reads as "no constraints, give me whatever you've got." For the
   eltwise-add demo that left function arguments as row-major bf16 in
   DRAM (`memref<64x32xbf16, #dram>`), while the op's *result* was
   inferred as TILE (`memref<2x1x!ttcore.tile<32x32, bf16>, #dram>`).
   The tt-lang JIT emits one common kernel shape: the reader uses
   `noc_async_read_tile(idx, accessor, addr)`, the compute thread
   treats each page as a 32x32 tile (4 16x16 faces), and the writer
   uses `noc_async_write_tile`. Feed it a row-major buffer and the
   page-byte arithmetic still lands inside the tensor, but the bytes
   it picks up are *not* a valid tile — for our 64x32 demo this
   manifested as the first ~6 rows of each tile being correct and the
   rest being zero (the first face overlaps row-major rows for the
   topmost rows by coincidence).
   The fix lives in
   `third_party/tt-mlir/.../TTNN/IR/TTNNOps.td` on
   `TTNN_TtLangOp`: register a `Layout::Tile` workaround on every
   input and result so the layout pass inserts `ttnn.to_layout` on
   row-major inputs and on the function-return path. (The op-aware
   `createEmptyTTNNOperandsWorkarounds(op)` overload pre-fills one
   slot per ranked-tensor operand; we use the zero-arg form and
   append, otherwise the slot count doubles and the verifier
   rejects the op.)

7. **`ttnn.tt_lang_op` and the deallocate pass.** Our flatbuffer
   emitter `TTNNToFlatbuffer.cpp` aliases the kernel's SSA result to
   the "out"-roled operand's `TensorRef` so the runtime's
   `gatherOutputTensors()` (keyed by `global_id`) finds the kernel's
   output at program end. The default `TTNNDeallocatePass` would
   otherwise see the "out" operand as a normal SSA value, mark the
   tt-lang op as its last use, and insert `ttnn.deallocate(%arg_out)`.
   The runtime would then erase the global_id from the tensor pool and
   the subsequent `return` call would FATAL with "Tensor not found in
   tensor pool."

   The fix lives in
   `third_party/tt-mlir/.../Dialect/TTNN/IR/TTNNOps.cpp` and
   `Dialect/TTNN/Transforms/Passes.cpp`. `TtLangOp` implements
   `MemoryEffectOpInterface::getEffects` by walking `arg_roles`:
   each `"in"` operand is published with `MemoryEffects::Read` on its
   `OpOperand &`, and each `"out"` operand with `MemoryEffects::Write`.
   The deallocate pass then calls
   `checkAndInsertDeallocation` -> `isWrittenByLastUser(value, lastOp)`
   which is a generic
   `MemoryEffectOpInterface::getEffectOnValue<MemoryEffects::Write>(value)`
   query. Any op that writes through one of its operands (today: the
   tt-lang kernel, the `update_cache` / `fill_cache` family) picks up
   this handling automatically, no name-keyed special case required.

   Per-operand `Read`/`Write` effects also let `arg_roles` be in any
   order -- for example `"in,out,in"` is a perfectly valid kernel
   signature, the verifier accepts it (it only requires one `"out"`
   per result), and the deallocate pass behaves correctly because it
   asks the interface about a specific `Value`, not about positional
   conventions. The kernel author's parameter order is preserved
   end-to-end, which matters because `@ttl.operation`'s generated
   reader/writer C++ references operand positions verbatim.

   The result-side of the alias is what the program returns and is
   managed by the standard return-value path, so this only sacrifices
   a single potentially-redundant dealloc; nothing leaks. The
   generic-interface contract is locked in by
   `test/ttmlir/Dialect/TTNN/Transforms/ttnn_deallocate_tt_lang_op.mlir`,
   which exercises an interleaved `arg_roles = "in,out,in"`.

### 5. Caching (only if measured)

Today there is no cache. The PJRT plugin already caches whole compiled
executables; tt-lang should cache its own compilations. Only if profiling
shows our resolve callback firing multiple times for the same arguments
should we add caching here -- and then it can be a single
`functools.lru_cache` on `resolve_kernel`, not a separate subsystem.

### 6. Autograd (if needed)

`torch.ops.tt.tt_lang_op` currently raises from its autograd hook to
prevent silent zero-gradient bugs. When backward support is required,
the canonical pattern is to author a separate `@tt_torch.kernel` (with
its own `kernel_id`) for the backward and have the autograd hook look it
up and invoke it.

## Why this shape

A few invariants kept the design honest:

* The custom op must stay inside the XLA graph end to end so that Shardy,
  layout, and ttnn.generic placement all compose with it. Anything that
  routes around the graph (e.g. a host callback) defeats the integration.
* The Python side owns only what it must own: the user-facing API, a
  `kernel_id` registry, and a single resolve entry point. Compile
  caching, ABI mirrors, and backend selection are deferred until there is
  real code to integrate them with.
* The `kernel_id` is the only stable identifier crossing the C / Python
  boundary; everything else (shapes, dtypes, mesh) is normal PJRT runtime
  data the plugin already understands.

## Where to look in the code

```
python_package/tt_torch/
├── custom_ops.py           # tt::tt_lang_op definition + fake + autograd
└── tt_lang.py              # decorator + registry + resolve_kernel
                            # + _serialize_compiled_kernel JSON producer
                            # (_ARTIFACT_FORMAT_VERSION is the schema gate)

pjrt_implementation/inc/api/module_builder/
├── module_builder.h        # ModuleBuilder::resolveTtLangKernels declaration
└── tt_lang_bridge.h        # tt_lang_bridge::resolveKernels declaration

pjrt_implementation/src/api/module_builder/
├── module_builder.cc       # ModuleBuilder::resolveTtLangKernels (delegator)
└── tt_lang_bridge.cc       # embedded pybind11 -> tt_torch.tt_lang.resolve_kernel
                            # (layout passthrough lives here)

tests/torch/ops/
└── test_tt_lang_kernel.py  # unit tests for the Python surface,
                            # incl. fake-`ttl` driver + real-tt-lang gate

docs/src/
└── tt_lang_integration.md  # this file
```

In the pinned tt-mlir tree (`jackzhang/tt_lang_integration`):

```
include/ttmlir/Dialect/TT{IR,NN}/IR/TT{IR,NN}Ops.td
                            # ttir.tt_lang_op / ttnn.tt_lang_op definitions

lib/Conversion/StableHLOToTTIR/StableHLOToTTIRPatterns.cpp
                            # stablehlo.custom_call @tt.tt_lang_op -> ttir.tt_lang_op

lib/Conversion/TTIRToTTNN/TTIRToTTNN.cpp
                            # ttir.tt_lang_op -> ttnn.tt_lang_op (kernel_artifact empty)

lib/Target/TTNN/TTNNToFlatbuffer.cpp
                            # createOp(TtLangOp) emits GenericOp + ProgramDescriptor
                            # from the JSON kernel_artifact (T4)

test/ttmlir/Conversion/StableHLOToTTIR/tt_lang_op.mlir
test/ttmlir/Dialect/TTNN/tt_lang_op.mlir
test/ttmlir/Silicon/TTNN/n150/tt_lang_op/tt_lang_op_flatbuffer.mlir
                            # lit tests for each lowering stage
```
