# Handoff — multi-output `tt_lang_op` result aliasing (blocks multi-output kernels on tt-xla)

> **STATUS: RESOLVED (2026-07-08).** The bug is fixed and verified end-to-end. Jump to
> [Resolution](#resolution) for what was actually wrong (it was **not** the runtime insert this
> doc originally hypothesised) and the one-file compiler fix. The original investigation notes
> below are kept for context but their root-cause conclusion (§"Root cause") is **superseded** by
> the Resolution section.

## Resolution

**Root cause (corrected): the two `"out"` operands collapse onto a single device buffer *before*
the kernel ever runs — a compile-time aliasing bug, not a runtime write-back bug.** Three separate
mechanisms pile on, each merging the two `torch.empty_like()` output placeholders:

1. **XLA (frontend)** CSEs the two identical uninitialised `torch.empty_like` tensors into one
   `stablehlo.constant` (`%cst`), passed as *both* out operands of the custom_call. (Seen directly
   in the incoming SHLO: `@tt.tt_lang_op(..., %cst, %cst)`.)
2. **tt-mlir CSE** merges the resulting `ttnn.full` / `ttnn.empty` ops (neither carries a memory
   effect, so CSE treats them as pure and dedups).
3. **Const-eval hoisting** pulls the shared zero-init into one `main_const_eval_0` whose two
   `ttcore.load_cached(@main_const_eval_0, [])` call sites the **runtime memoises by
   (function, args)** — so both outputs resolve to the *same* cached tensor. This is the layer the
   original "runtime" symptom pointed at, but the fix belongs upstream.

The kernel then writes both of its outputs into one DRAM buffer and the last write wins — every
result reads back identical (`di == dgc`). The runtime's `insertTTNNTensorAndValidate(... size-1)`
line is a **red herring**: with distinct output buffers, both outputs are written in place and
surfaced correctly with no runtime change (verified — see below).

**Fix (one file):** `lib/Dialect/TTNN/Transforms/TTNNLowerTTLangToGeneric.cpp`. This is the last
pass before flatbuffer emission (no CSE / const-eval runs after it). When lowering
`ttnn.tt_lang_op` → `ttnn.generic`, rebuild **every** `"out"` init operand as a fresh `ttnn.empty`
of the operand's type. `ttnn.empty` is the correct op for a write-only DPS destination
(uninitialised, no wasteful zero-fill), it carries `TTCoreNonCacheableTrait` so const-eval never
hoists/dedups it, and creating one op per output keeps the buffers distinct with nothing downstream
to merge them back. (An earlier attempt in `StableHLOToTTIRPatterns.cpp` using `ttir.empty` was
reverted: `ttir.empty` survives to TTNN but the later `ttnn.empty` still gets CSE'd + const-eval
memoised, so it did not hold.)

**Verification** (all on device, `TT_VISIBLE_DEVICES=0`):
- Minimal repro below: `di != dgc`, `di` PCC 0.99999 vs analytic `dL_dinput`, `dgc` PCC 0.99999.
- Single norm / 3-norm chain / 3-D input through autograd: all grads PCC ≥ 0.9999.
- Single-output eltwise e2e (`examples/pytorch/tt_lang_eltwise_add.py`): PCC 0.99999 (no regression).
- Full `custom_rmsnorm_override.py --xla`: **all grads PCC ≥ 0.99998, gate passes.**

**Second, independent finding — the `--xla` gate itself was degenerate.** `RefRMSNorm` initialises
`weight = 1`, which combined with the `square().mean()` loss on the *final* norm makes that norm
scale-invariant: RMSNorm's Jacobian nulls the radial direction the loss gradient points along, so
the gradient to every upstream parameter vanishes to ~1e-9 (numerical noise). PCC between two noise
vectors is meaningless, so the gate failed *even for the fully-native reference model* (only
`head.weight`, the one non-vanishing grad, matched). Fixed by giving the norm weights non-trivial
values in `_build_pair` (`custom_rmsnorm_override.py`) so every parameter carries a real,
comparable gradient. This is unrelated to the kernel/compiler and was masking the true (now
passing) result.

---

Self-contained handoff for the one remaining blocker to running the custom RMSNorm backward
kernel end-to-end in the XLA graph. Everything else in this demo is fixed and verified; see
`FINDINGS.md` for the full trail. This doc is scoped to the **multi-output execution bug**.

## TL;DR
A `tt_lang_op` (custom tt-lang kernel) with **two or more `"out"` operands** produces the
**same value for every result** on device — only the *last* out operand is written back. The
runtime's generic-op handler registers a single output tensor against `io_tensors[size-1]` and
drops the rest. Single-output tt-lang ops (the only case the eltwise e2e test exercises) work
fine. Our RMSNorm-backward kernel needs **two** outputs (`dL_dinput`, `dL_dgamma_comp`), so it
hits this.

The kernel itself is correct (native ttl/ttnn PCC = 1.0 for both outputs); this is purely a
tt-mlir **runtime** limitation.

## Symptom
`python custom_rmsnorm_override.py --xla` fails the PCC gate with `head.weight` PCC 0.99999 but
`input` and all block grads ≈ 0, and the numbers are **byte-identical across any kernel edit**
(not caching — a fresh `TT_METAL_CACHE` recompiles 1682 files and resolve emits a new 28444-byte
artifact each run).

## Minimal reproduction (no autograd, no model)
```python
# cd tt-xla && source venv/activate ; run under TT_VISIBLE_DEVICES=0
import torch, torch_xla, tt_torch
import torch_xla.core.xla_model as xm
import custom_rmsnorm_override as ovr      # registers the 2-out tt_lang_op

T, C = 128, 128
x = torch.randn(T, C, dtype=torch.bfloat16); g = torch.randn(1, C, dtype=torch.bfloat16)
rms = torch.randn(T, 1, dtype=torch.bfloat16).abs() + 0.5; dL = torch.randn(T, C, dtype=torch.bfloat16)
dev = xm.xla_device()
di  = torch.empty_like(x.to(dev))          # out 0: dL_dinput
dgc = torch.empty_like(x.to(dev))          # out 1: dL_dgamma_comp
ovr.rmsnorm_bw_op(x.to(dev), g.to(dev), rms.to(dev), dL.to(dev), di, dgc)
xm.mark_step()
# BUG: di == dgc exactly (PCC 1.0, 100% element match). di never receives out-0.
print((di.cpu() == dgc.cpu()).float().mean())   # -> 1.0
```
Reading the wrapper's functional return instead (`ret = ovr.rmsnorm_bw_op(...)`, use `ret[0]`,
`ret[1]`) trips a runtime assert:
```
LOG_ASSERT ../runtime/lib/ttnn/types/types.cpp:111: it != liveTensors.end()
    "Tensor not found in tensor pool"
```
Both symptoms are the same underlying cause: out-0's tensor is never inserted into the pool.

## Root cause (traced through the stack)
The lowering path is correct end to end; the runtime is not.

1. **Python** (`python_package/tt_torch/custom_ops.py`): `tt_lang_op` emits a
   `stablehlo.custom_call` with N results (tuple), mutation-style — the wrapper does
   `tensors[out_idx].copy_(result_i)` for each out operand. ✅
2. **DecomposeCustomCallTuples** (`.../StableHLO/Transforms/DecomposeCustomCallTuples.cpp`,
   extended for this demo): splits the tuple-returning custom_call into a multi-result op so
   Shardy can run. ✅ (verified: result 1 works)
3. **StableHLOToTTIR** (`StableHLOToTTIRPatterns.cpp`, `StableHLOTTLangOpConversionPattern`):
   builds `ttir.tt_lang_op` with one result per tuple element. ✅ (`TTIR_TTLangOp` /
   `TTNN_TTLangOp` are `Variadic<AnyRankedTensor>:$results`, verifier enforces `#out ==
   #results`).
4. **TTNNLowerTTLangToGeneric** (`.../TTNN/Transforms/TTNNLowerTTLangToGeneric.cpp:439-458`):
   correctly ties result `r` to out operand `numIns + r` and replaces each result use with its
   tied out operand before erasing the op:
   ```cpp
   for (unsigned r = 0; r < numResults; ++r)
     op.getResult(r).replaceAllUsesWith(op.getInputs()[numIns + r]);
   ```
   So downstream references each out operand directly. ✅
5. **TTNNToFlatbuffer**: emits a `GenericOp` whose `io_tensors` = all operands (ins then outs);
   for us `io_tensors` has 6 entries, outs at indices 4 (`dL_dinput`) and 5 (`dL_dgamma_comp`).
6. **Runtime — THE BUG** (`runtime/lib/ttnn/operations/generic/generic_op.cpp:506-509`, and the
   mesh variant `522-525`):
   ```cpp
   ::ttnn::Tensor outputTensor = ::ttnn::generic_op(ioTensors, *programDescriptor);
   tensorPool.insertTTNNTensorAndValidate(
       op->io_tensors()->Get(ioTensorsSize - 1), outputTensor);   // only the LAST out!
   ```
   `::ttnn::generic_op` returns a **single** tensor, and the runtime registers it against only
   `io_tensors[ioTensorsSize - 1]`. The other out operand(s) — here `io_tensors[4]`
   (`dL_dinput`) — are never inserted into `liveTensors`. Any later reference to out-0 therefore
   finds nothing in the pool (`types.cpp:111` assert), or, via the mutation `copy_`, resolves to
   the only registered out tensor (out-1 = `dL_dgamma`), which is why `di == dgc`.

The eltwise e2e test (`tests/torch/ops/test_tt_lang_kernel_e2e.py`) only ever has **one** out
operand, so `ioTensorsSize - 1` happens to be correct there and the bug was never seen.

## Fix direction
The runtime needs to register **every** `"out"` io_tensor with its on-device result, not just the
last. Key questions/steps for whoever picks this up:

1. **Does `::ttnn::generic_op` expose all outputs?** It currently returns one `::ttnn::Tensor`.
   Check tt-metal for a multi-output overload / a way to get each out operand's `Buffer*` after
   launch. The kernel's `dm_write` already writes both out DRAM buffers on device; the runtime
   just isn't surfacing them.
2. **How many outs are there at runtime?** The generic-op flatbuffer needs to carry the out
   count (or the ins/outs split). `ttnn.tt_lang_op`'s `arg_roles` (`in* out+`) and the DPS
   result count are known at lowering time (`TTNNLowerTTLangToGeneric.cpp` computes `numResults`
   / `numIns`); thread that through to the `GenericOp` record if it isn't already, then in
   `generic_op.cpp` loop over the trailing `numResults` io_tensors and
   `insertTTNNTensorAndValidate` each against its corresponding output tensor/buffer.
3. **Mesh path**: apply the same fix to the `MeshProgramDescriptor` branch (`522-525`).
4. **Verify** with the minimal repro above (expect `di != dgc`, `di` ≈ analytic `dL_dinput`), and
   then the full `custom_rmsnorm_override.py --xla` should pass the PCC gate (native kernel
   already gives PCC 1.0 for both outputs).

Rebuild after the change (the plugin dynamically loads the installed shared lib, so no plugin
relink needed):
```bash
cd tt-xla && source venv/activate
cd third_party/tt-mlir/src/tt-mlir/build
export TT_METAL_RUNTIME_ROOT="$TT_MLIR_HOME/third_party/tt-metal/src/tt-metal"
cmake --build .                              # relinks libTTMLIRRuntime.so / libTTMLIRCompiler.so
cmake --install . --component SharedLib
cmake --install . --component DistributedRuntime   # runtime lib
```
(The runtime lives in `libTTMLIRRuntime.so`; confirm which component installs it and that the
install dir `third_party/tt-mlir/install/lib` updates.)

## Context — everything else is already fixed (see FINDINGS.md)
- Shardy tuple blocker → extended `DecomposeCustomCallTuplesPass` (tt-mlir, rebuilt).
- ttl spec-0.17 porting (fill/broadcast namespace, tuple capture, bf16).
- resolve compile-capture cache bug → patch `_make_cache_key` in `tt_torch/tt_lang.py`.
- cb_configs `dfb_index` serialization → `_serialize_cb_config` in `tt_torch/tt_lang.py`.
- kernel `dL_dinput` math (gamma row-broadcast via `ones @ gv`) → native PCC 1.0.

## Reproduction assets in the repo
- `custom_rmsnorm_override.py` — the override + `--xla` runner (2-out kernel).
- `ttl_rmsnorm_bw_2pass.py` — the ported+fixed kernel.
- `kernel_native_check.py` — native ttl/ttnn check (bypasses tt-xla): both outputs PCC ~1.0,
  proving the kernel is correct and isolating the bug to the tt-xla runtime.
