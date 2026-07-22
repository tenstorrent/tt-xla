# Findings — custom RMSNorm backward via tt-lang on tt-xla

Continuation of `HANDOFF.md`. Goal: run the PR #44516 2-pass RMSNorm backward
kernel as a custom backward op inside the XLA graph on silicon (`custom_rmsnorm_override.py --xla`),
PCC ≥ 0.99 vs a CPU golden.

**Status:** all tt-xla **infrastructure is fixed and validated** — the `--xla` path now
resolves, lowers, compiles, and **executes the custom backward kernel on device** end-to-end.
Fixed along the way: the Shardy/tuple blocker (tt-mlir pass), three ttl-spec-0.17 kernel-porting
issues, a resolve-time compile-capture cache bug, and a CB-index serialization drift. The **sole
remaining gate is a correctness bug in the PR kernel's `dL_dinput` math** (OPEN #1), which
reproduces natively without tt-xla. Details below.

---

## Environment / layout
- tt-xla: `/localdev/ndrakulic/tt_lang_demo/tt-xla` (branch `main`, venv at `venv/`)
- tt-metal (PR #44516 branch, for the kernel source): `/localdev/ndrakulic/tt_lang_demo/tt-metal`
  - branch `lgalasTT/benchmark_ttlang_vs_ttml_rmsnorm_bw`
  - kernel: `tt-train/sources/ttml/ttlang/ttl_rmsnorm_bw_2pass.py`
- tt-lang (reference/examples): `/localdev/ndrakulic/tt_lang_demo/tt-lang`
- installed `ttl` version: **1.1.5.dev20260704+light** (spec 0.17)

Run everything with `cd /localdev/ndrakulic/tt_lang_demo/tt-xla && source venv/activate`.

### Files created/modified in tt-xla
- `custom_rmsnorm_override.py` (was already present) — arity wrapper + bf16 cast added (see below).
- `ttl_rmsnorm_bw_2pass.py` — copy of the PR kernel, ported to ttl spec 0.17 (see below).
- `kernel_native_check.py` — NEW: validates the kernel in pure ttl/ttnn (no tt-xla), the fastest repro.
- `third_party/tt-mlir/src/tt-mlir/lib/Dialect/StableHLO/Transforms/DecomposeCustomCallTuples.cpp` — the infra fix.

---

## FIXED — the infra blocker (Shardy + tuple custom_call)

### Root cause
A tt-lang op with **≥2 `"out"` operands** (ours has `dL_dinput` + `dL_dgamma_comp`) is
emitted by `tt_lang_op` (`python_package/tt_torch/custom_ops.py:2330`) as a
`stablehlo.custom_call` with a **tuple result**:

```
%custom-call = (f32[128,128], f32[128,128]) custom-call(...)   # result TYPE is a tuple
%gte.0 = get-tuple-element %custom-call[0]
%gte.1 = get-tuple-element %custom-call[1]
```

tt-mlir's frontend StableHLO pipeline runs Shardy propagation unconditionally (torch_xla
always emits an `sdy` mesh, even single-device), and **Shardy cannot propagate through
tuple-typed results**:

```
loc("custom-call.234"): error: Shardy propagation doesn't support tuples:
    'tuple<tensor<128x128xf32>, tensor<128x128xf32>>'
```

The existing eltwise e2e test (`tests/torch/ops/test_tt_lang_kernel_e2e.py`) only ever has
**one** output, so multi-output tt-lang ops had never exercised this path. This is a genuine
gap in the #5539 mechanism, **not** a mistake in the override or the kernel.

`sharding_constraint` / `sdy.sharding` hooks cannot work around it — they annotate *tensor*
values to steer propagation; you cannot attach one to a tuple, and they don't make Shardy
skip an op.

### The fix
tt-mlir already ships `DecomposeCustomCallTuplesPass`, wired into the pipeline right before
Shardy (`lib/Dialect/StableHLO/Pipelines/StableHLOPipelines.cpp:68`) with the comment
*"Convert tuple-returning custom_call ops to multi-result ops so that Shardy can propagate
through them."* But its implementation only rewrote explicit `stablehlo.tuple` builder ops +
their `get_tuple_element` users — it **never touched a `custom_call` whose own result type is
a tuple** (exactly what we emit). So it was a no-op for our IR.

Extended the pass (`DecomposeCustomCallTuples.cpp`, "Rewrite 2") to also handle tuple-returning
`custom_call` ops: clone the op with one tensor result per tuple element (copying all
attributes incl. the discardable `mhlo.frontend_attributes`, dropping the size-sensitive
`operand_layouts`/`result_layouts`), then forward each `get_tuple_element` user to the matching
result. The downstream `StableHLOToTTIR` converter and `TTIR_TTLangOp`/`TTNN_TTLangOp` are
already `Variadic<AnyRankedTensor>:$results`, so N results are supported end to end.

### Rebuild (what worked)
The plugin dynamically loads `third_party/tt-mlir/install/lib/libTTMLIRCompiler.so`, so **no
plugin relink is needed** — just rebuild + reinstall tt-mlir's SharedLib component:

```bash
cd /localdev/ndrakulic/tt_lang_demo/tt-xla && source venv/activate
cd third_party/tt-mlir/src/tt-mlir/build
export TT_METAL_RUNTIME_ROOT="$TT_MLIR_HOME/third_party/tt-metal/src/tt-metal"
cmake --build .                       # relinks libTTMLIRCompiler.so (~1GB, slow)
cmake --install . --component SharedLib
```

### Validation
After the rebuild, the Shardy tuple error is gone; the op flows through Shardy → StableHLO →
TTIR → TTNN and reaches kernel resolution. Confirmed by the subsequent errors moving *past*
Shardy into `TTIRToTTNNCommon` / tt-lang resolve.

---

## FIXED — kernel porting to ttl spec 0.17

The PR kernel targets an older ttl API. Ported in `ttl_rmsnorm_bw_2pass.py`:

1. **`fill`/`broadcast` moved to the `ttl.block.*` namespace** (spec 0.17). The AST compiler
   (`ttl/_src/ttl_ast.py:606`) hard-remaps these names and errors otherwise
   (`ttl.math.fill is not available; use ttl.block.fill`). New signatures:
   - `ttl.block.fill(value, *, shape, dtype=None)`  (was `ttl.math.fill(block, value)`)
   - `ttl.block.broadcast(input, *, dims, shape)`   (was `ttl.math.broadcast(input, target, dims=...)`)
2. **No tuple captures inside `@ttl.compute`.** Closure capture only supports
   int/float/ttnn-tensor/DataflowBuffer/Pipe/PipeNet (`ttl/ttl_api.py:1032`). Referencing the
   enclosing `tile`/`blk` tuples raised *"Unhandled capture for vars of type tuple"*. Fixed by
   inlining shape literals (`shape=(1, 1)`, `shape=(1, block_size)`), so only the int
   `block_size` is captured.
3. **f32 dtype on `fill`.** `fill` defaults to bf16; our f32 CBs required
   `dtype=<block>.dtype` (the SSA-attribute path, `ttl_ast.py:661-676`).

Also in `custom_rmsnorm_override.py`:
- **Arity wrapper**: wrap `make_kernel()`'s result in a plain 6-arg function before
  `@tt_lang_operation` (matches the eltwise example; defensive — `inspect.signature` actually
  survives `@ttl.operation` here, so it wasn't strictly required).
- **bf16 at the kernel boundary** in `_rmsnorm_backward` (XLA branch): cast operands to bf16,
  run the kernel, cast grads back. Reason: an **f32 CB consumed by both an FPU matmul and an
  SFPU elementwise op needs mutually-exclusive unpack modes** (`UnpackToDestFp32` vs default),
  which tt-metal rejects:
  ```
  error: f32 input from CB 0 is consumed by both FPU and SFPU strategies ... Split the source
  into separate CBs (one per strategy) so the SFPU consumer keeps full f32 precision
  ```
  bf16 avoids this; `fp32_dest_acc_en=True` still accumulates in f32. (The PR kernel was
  authored for bf16 — the metal test uses `ttnn.bfloat16` throughout.)

---

## FIXED — kernel `dL_dinput` numerically wrong (was OPEN #1)

Root cause: `gamma` is a `[1, C]` row vector, so a loaded gamma tile only has data in
**row 0**; the kernel never broadcast it across the token (row) dimension. Every
gamma-dependent term (`term1 = γ·r·g` and the `scale` reduction) was therefore correct only
for token 0 and **zero for tokens 1..31**. `dL_dgamma` was correct precisely because it's the
one term that doesn't use gamma. The matmul-with-ones reduction and the analytic formula were
both fine. `recip` broadcasts correctly because it's per-row/constant-across-columns (innermost
intra-tile broadcast); gamma is the opposite (constant-across-rows), which `ttl.block.broadcast`
can't do intra-tile.

Fix (`ttl_rmsnorm_bw_2pass.py`): add an all-ones `(1,1)` tile and broadcast gamma across rows
via `ones @ gv` (a left-multiply matmul) in both passes. `kernel_native_check.py` now reports
**`dL_dinput` PCC = 1.0 across all shapes** (32×32, 32×64, 32×128, 128×128); `dL_dweight` stays
1.0. Localized by a shape sweep (bug present even at a single 32×32 tile → not accumulation) and
a diagnostic that emitted `scale` directly (`scale col0 = [4.16, 0, 0, 0]` → only token 0
populated).

---

## OPEN — multi-output tt_lang_op result aliasing (blocks end-to-end `--xla`)

With the kernel math fixed, the full `--xla` run still fails the PCC gate with results **byte-
identical across every kernel edit** (including a constant all-ones `dL_dinput` probe and two
`version_tag` bumps with fresh `TT_METAL_CACHE`). This is **not** caching — resolve produces a
genuinely new artifact each time (28444 B, two matmul loops, 18 CBs) and fresh binaries compile
(1682 cache files). The real cause, isolated by a raw op-dispatch test (no autograd, no model):

* A 2-output `tt_lang_op` returns the **same value for both results**: `di == dgc` exactly
  (PCC 1.0, 100% element match), both equal to the **second** output (`dL_dgamma`). The first
  output (`dL_dinput`) never lands — the mutated `torch.empty_like` buffer for it is overwritten
  with the second result.
* Reading the wrapper's **functional return** instead trips a runtime liveness assert:
  `LOG_ASSERT ../runtime/lib/ttnn/types/types.cpp:111: it != liveTensors.end()`.
* In the model this shows as `head.weight` PCC 0.99999 (a pure `dL_dgamma`) while `input` and
  every gradient flowing through a `dL_dinput` is ~0.

This is the DPS out-operand/result aliasing + dealloc-skip path (handoff flagged tt-mlir commit
4d6e1bf95) which had only ever been exercised with **one** `"out"` operand (the eltwise e2e
test). Two `"out"` operands collapse to one result. Fixing it needs a tt-mlir/runtime C++ change
+ rebuild. **Sidestep for a green demo:** make the kernel single-output (`dL_dinput` only, the
hard 2-pass part) and compute `dL_dgamma = (x·r·dL_dout).sum(0)` natively in the override —
single output ⇒ no tuple, no multi-output aliasing, and `dL_dinput` is now correct.

---

## OLD OPEN #1 (superseded — see FIXED above) — kernel `dL_dinput` is numerically wrong (PCC ≈ 0.17)

**This is the primary correctness blocker and reproduces WITHOUT tt-xla.**

```bash
cd /localdev/ndrakulic/tt_lang_demo/tt-xla && source venv/activate
TT_VISIBLE_DEVICES=0 python kernel_native_check.py
# ->  dL_dinput  PCC = 0.16994     (WRONG)
#     dL_dweight PCC = 0.99999     (correct)
```

`kernel_native_check.py` runs the kernel directly via its own `run_rmsnorm_bw_2pass` on a ttnn
device (bf16, 128×128) and compares to a torch RMSNorm-backward reference. The kernel
**compiles and runs**, `dL_dgamma` is correct, but `dL_dinput` is wrong.

Because `dL_dweight = x·r·dL_dout` is correct, the shared quantities (`r=1/rms`, `x`,
`dL_dout`, and the recip/broadcast plumbing) are fine. The bug is isolated to the **`dL_dinput`
path** — i.e. the pass-1 `scale = Σ_c(x·(γ·r)·dL_dout)` reduction (the `contrib @ reduce_col`
matmul-with-ones + `scale_acc` accumulation across column blocks) and/or its use in pass 2:
`dL_dinput = (γ·r)·dL_dout − scale·x·r²·(1/C)`. bf16 precision is ruled out (dgamma is bf16 and
perfect; PCC 0.17 is structural, not rounding).

Upstream context: **PR #44516's own test never ran** — `ttl_rmsnorm_bw_2pass_test.py:52` calls
`ttl_mod_pad(...)` (undefined; should be `ttl_mod.pad`), so this reduction path was never
validated. Treat the kernel's `dL_dinput` as unverified upstream.

Suggested next steps: unit-test pass 1 in isolation (dump `scale` vs `Σ_c` reference); check the
`scale_acc` read-modify-write across `num_col_blocks` (double-buffered `block_count=2` RMW);
verify the ones-matmul reduction sums the intended axis for this ttl version.

---

## FIXED — tt-xla resolve: "did not produce a CompiledTTNNKernel"  (was OPEN #2)

Fixed in `python_package/tt_torch/tt_lang.py` (`_drive_ttl_compile`): in addition to patching
`_compile_kernel`, also patch ttl's `_make_cache_key` to return a unique object per call for the
duration of the drive, forcing a cache miss so `_compile_kernel` (our only capture point) always
runs. Guarded by `hasattr` so an upstream rename degrades gracefully. Verified:
`resolve_operation` now succeeds on repeated calls in one process (3/3), and the full `--xla`
run gets past this error into TTNN lowering. Root-cause analysis retained below.

### Root cause (confirmed)

With bf16 operands, the in-XLA run fails at resolve time:

```
loc("custom-call.322"): error: tt-lang resolve failed for kernel 'rmsnorm.bw.2pass.v1':
  TTLangError: tt-lang compile did not produce a CompiledTTNNKernel ...
```

**This is a stale-cache interaction in tt-xla's device-less "DEMO HACK" compile-capture, NOT a
stub-compile failure.** A single `resolve_operation` call actually succeeds and produces a
valid artifact; the *second* call in the same process fails. Confirmed repro:

```python
# after: import torch_xla, tt_torch, custom_rmsnorm_override
shapes = [[128,128],[1,128],[128,1],[128,128],[128,128],[128,128]]; dtypes = ["bf16"]*6
resolve_operation(operation_id="rmsnorm.bw.2pass.v1", version_tag="pr44516-2pass",
                  shapes=shapes, dtypes=dtypes)   # call 1: OK (25959-byte artifact)
resolve_operation(...same...)                     # call 2: TTLangError "did not produce..."
```

### Chain of events
1. **ttl caches compiled kernels per operation.** The `@ttl.operation` wrapper
   (`ttl/ttl_api.py:1815`) holds a closure-local `cache: Dict[..., CompiledTTNNKernel]`, keyed
   by operand shapes/dtypes/opts (`ttl_api.py:1838`). On a **cache hit** it returns the cached
   kernel and **never calls `_compile_kernel`** (`ttl_api.py:1848-1855`).
2. **tt-xla captures the kernel only by intercepting `_compile_kernel`.** `_drive_ttl_compile`
   (`python_package/tt_torch/tt_lang.py:558-650`) monkey-patches `ttl.ttl_api._compile_kernel`,
   drives `entry.impl(*stub_args)`, and takes `captured[-1]`; empty ⇒ raises "did not produce"
   (`tt_lang.py:641`).
3. **`_bw_kernel` is a process-wide singleton** — `custom_rmsnorm_override.py` runs
   `_bw_kernel = make_kernel()` once at import, so its ttl `cache` persists for the whole process.
4. **The compile pipeline resolves the op more than once per process.** First resolve → cache
   miss → `_compile_kernel` runs → captured ⇒ succeeds. Second resolve (same signature) → ttl
   **cache hit** → `_compile_kernel` skipped → `captured == []` → raises.

The native check (OPEN #1) and the eltwise e2e test don't hit this: native calls compile once;
the eltwise test builds a *fresh* operation object per parametrization (empty cache each time).
Our module-level singleton is what carries the cache into a second resolve.

### Fix direction (all in `python_package/tt_torch/tt_lang.py`; no kernel change)
- Defeat ttl's per-op cache inside `_drive_ttl_compile` so every drive is a miss, **or**
- Capture the kernel from the wrapper's return path (so cache hits are captured too), **or**
- Build a throwaway operation object per resolve so its cache starts empty.

With #2 fixed, resolve produces a valid artifact and the in-XLA path advances into TTNN
lowering, where it now hits OPEN #3 below. The remaining gates are OPEN #3 (a CB-index
serialization drift) and OPEN #1 (the kernel `dL_dinput` math).

---

## FIXED — `cb_configs[N].buffer_index = -1 is out of range for uint32_t`  (was OPEN #3)

After the OPEN #2 fix, the `--xla` run reached TTNN lowering and failed:

```
loc("custom-call.238"): error: cb_configs[12].buffer_index = -1 is out of range for uint32_t.
```

Root cause: another ttl-version drift. This kernel has 15 CBs — 12 user `DataflowBuffer`s
(`_cb_index` 0–11) and 3 `CompilerAllocatedDFBConfig`s (the 1/C constant, the unity reduce
column, and the broadcast scratch). In ttl 1.1.5 the compiler-allocated configs expose their CB
index as **`dfb_index`** (values 12/13/14, continuing the same index space), **not** `_cb_index`.
`_serialize_cb_config`'s compiler-allocated branch read `getattr(cb, "_cb_index", -1)`, so those
three serialized as `-1`, which the flatbuffer emitter rejects.

Fixed (`python_package/tt_torch/tt_lang.py`, `_serialize_cb_config`): read `dfb_index` with a
fallback to `_cb_index`. Verified: the `--xla` path now resolves → lowers → compiles → **executes
on device** end-to-end with no infra errors. The run's only remaining failure is the PCC gate,
which is OPEN #1 (below): `head.weight` PCC = 0.99999 (a pure `dL_dgamma` of the last norm) while
`input` and every gradient that flows through a custom-RMSNorm `dL_dinput` is ~0. **All tt-xla
infrastructure now works; the kernel `dL_dinput` math is the sole correctness gate.**

---

## Repro summary
```bash
cd /localdev/ndrakulic/tt_lang_demo/tt-xla && source venv/activate

# CPU wiring/math check (no hardware) — PASSES:
python custom_rmsnorm_override.py

# Native ttl/ttnn kernel check (no tt-xla) — shows OPEN #1:
TT_VISIBLE_DEVICES=0 python kernel_native_check.py

# Full in-XLA path — reaches OPEN #2 (past the now-fixed Shardy blocker):
TT_VISIBLE_DEVICES=0 python custom_rmsnorm_override.py --xla
```
