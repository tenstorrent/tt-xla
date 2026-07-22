# tt-lang custom-backward on tt-xla — errors encountered & fixes

Consolidated log of every error hit while getting a multi-output tt-lang kernel
(the RMSNorm custom backward, and the minimal matmul stand-in) to resolve,
lower, compile, and execute inside the XLA graph on device. Each entry: the
symptom, the root cause, the fix, and the e2e test that now gates it.

All fixes are **verified on device** (`xla:0`). The e2e tests live in
`tests/torch/ops/test_tt_lang_kernel_e2e.py`.

---

## 1. Shardy propagation doesn't support tuples

- **Symptom**
  ```
  loc("custom-call.234"): error: Shardy propagation doesn't support tuples:
      'tuple<tensor<128x128xf32>, tensor<128x128xf32>>'
  ```
- **Root cause** — A tt-lang op with **≥2 `"out"` operands** is emitted as a
  `stablehlo.custom_call` with a **tuple result**. tt-mlir runs Shardy
  propagation unconditionally (torch_xla always emits an `sdy` mesh, even
  single-device), and Shardy cannot propagate through tuple-typed results. The
  single-output eltwise e2e test never exercised this; multi-output ops had
  never hit the path. `sharding_constraint` / `sdy.sharding` hooks can't help —
  they annotate tensor values, not tuples.
- **Fix** — Extended `DecomposeCustomCallTuplesPass`
  (`lib/Dialect/StableHLO/Transforms/DecomposeCustomCallTuples.cpp`) to also
  rewrite **tuple-returning `custom_call` ops** (previously it only handled
  explicit `stablehlo.tuple` builder ops): clone the op with one tensor result
  per tuple element, then forward each `get_tuple_element` user to the matching
  result. Downstream (`StableHLOToTTIR`, `TTIR_TTLangOp`/`TTNN_TTLangOp`) is
  already `Variadic<AnyRankedTensor>`.
- **Gated by** — `test_tt_lang_multi_output_e2e` (any ≥2-output op reaches this).

---

## 2. Multi-output result aliasing (both outputs read back identical)

- **Symptom** — A 2-output `tt_lang_op` returns the **same value for both
  results** (`out0 == out1`, PCC 1.0, 100% element match), both equal to the
  **last** output. Reading the functional return instead trips a runtime assert:
  ```
  LOG_ASSERT ../runtime/lib/ttnn/types/types.cpp:111: it != liveTensors.end()
      "Tensor not found in tensor pool"
  ```
  Byte-identical across every kernel edit (not caching).
- **Root cause** — The two `"out"` `torch.empty_like()` placeholders collapse
  onto a **single device buffer before the kernel runs** (a compile-time
  aliasing bug, not a runtime write-back bug). Three mechanisms pile on:
  1. XLA CSEs the two identical uninitialised `torch.empty_like` tensors into
     one `stablehlo.constant`, passed as *both* out operands.
  2. tt-mlir CSE merges the resulting `ttnn.full`/`ttnn.empty` (no memory
     effect → treated as pure).
  3. Const-eval hoisting pulls the shared zero-init into one `load_cached`
     whose call sites the runtime memoises by `(function, args)` → both outputs
     resolve to the same cached tensor.
  The kernel writes both outputs into one buffer; last write wins.
- **Fix** — `lib/Dialect/TTNN/Transforms/TTNNLowerTTLangToGeneric.cpp` (last
  pass before flatbuffer emission). When lowering `ttnn.tt_lang_op` →
  `ttnn.generic`, rebuild **every `"out"` init operand as a fresh `ttnn.empty`**
  of the operand's type. `ttnn.empty` is write-only DPS-correct, carries
  `TTCoreNonCacheableTrait` (const-eval never hoists/dedups it), and one op per
  output keeps buffers distinct with nothing downstream to merge them.
  *(An earlier `ttir.empty` attempt in `StableHLOToTTIRPatterns.cpp` was
  reverted — the later `ttnn.empty` still got CSE'd + const-eval memoised.)*
- **Gated by** — `test_tt_lang_multi_output_e2e`
  (`assert frac_equal < 0.5` catches the collapse; per-output PCC would not).

---

## 3. CB config `buffer_index = -1 out of range for uint32_t`

- **Symptom**
  ```
  loc("custom-call.238"): error: cb_configs[12].buffer_index = -1 is out of
      range for uint32_t.
  ```
- **Root cause** — ttl-version drift. Kernels with matmul/broadcast get
  **compiler-allocated** scratch CBs (`CompilerAllocatedDFBConfig`). In ttl
  1.1.5 these expose their CB index as **`dfb_index`** (continuing the CB-index
  space after the user `DataflowBuffer`s), **not** `_cb_index`.
  `_serialize_cb_config` read `getattr(cb, "_cb_index", -1)`, so every
  compiler-allocated CB serialized as `-1`, which the flatbuffer emitter rejects.
- **Fix** — `python_package/tt_torch/tt_lang.py`, `_serialize_cb_config`: read
  `dfb_index` with a fallback to `_cb_index`, so a genuinely missing index still
  surfaces as the emitter's range error rather than silently.
  ```python
  compiler_cb_index = getattr(cb, "dfb_index", None)
  if compiler_cb_index is None:
      compiler_cb_index = getattr(cb, "_cb_index", -1)
  ```
- **Gated by** — `test_tt_lang_multi_output_e2e` (the matmul emits a
  compiler-allocated CB; probed as `dfb_index=4, _cb_index=None`).

---

## 4. resolve failed: "did not produce a CompiledTTNNKernel"

- **Symptom**
  ```
  loc("custom-call.322"): error: tt-lang resolve failed for kernel
    'rmsnorm.bw.2pass.v1': TTLangError: tt-lang compile did not produce a
    CompiledTTNNKernel ...
  ```
  A single `resolve_operation` succeeds; the **second** call in the same process
  (same signature) fails.
- **Root cause** — tt-lang's `@ttl.operation` wrapper memoizes each compiled
  kernel in a closure-local cache; on a **cache hit** it returns the kernel
  **without calling `_compile_kernel`** (and in compile-only mode returns
  `None`). tt-xla captures the artifact **only** by intercepting
  `_compile_kernel`. Because operation objects are process-wide singletons, the
  compile pipeline resolving the same signature more than once (a model reusing
  a norm/matmul width) hits tt-lang's cache → nothing captured → raise.
- **Fix** — `python_package/tt_torch/tt_lang.py`, `_drive_ttl_compile`: mirror
  tt-lang's cache. On a fresh compile (capture non-empty) store the kernel in a
  process-level `_COMPILED_KERNEL_CACHE` keyed by
  `(operation_id, version_tag, operand-signature)`; when a drive captures
  nothing (tt-lang cache hit) return the kernel recorded for the same key.
- **Gated by** — `test_tt_lang_multi_output_reused_signature_e2e` (one op called
  twice with identical signature; verified the real `_compile_kernel` runs once
  while both resolves succeed).

---

## Coverage matrix

| # | Error | Layer | Fixed in | e2e test gate |
|---|-------|-------|----------|---------------|
| 1 | Shardy tuple | tt-mlir | `DecomposeCustomCallTuples.cpp` | `test_tt_lang_multi_output_e2e` |
| 2 | Multi-output aliasing | tt-mlir | `TTNNLowerTTLangToGeneric.cpp` | `test_tt_lang_multi_output_e2e` |
| 3 | `dfb_index` CB `-1` | tt-xla Py | `tt_lang.py::_serialize_cb_config` | `test_tt_lang_multi_output_e2e` |
| 4 | resolve cache miss | tt-xla Py | `tt_lang.py::_drive_ttl_compile` | `test_tt_lang_multi_output_reused_signature_e2e` |
