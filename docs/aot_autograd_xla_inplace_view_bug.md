# AOTAutograd + XLA: `FunctionalTensorMode` inplace_view metadata fixup bug

## Summary

When using `aot_autograd()` as part of a custom `torch.compile` backend on XLA
tensors, models that contain ops tagged as `inplace_view` (e.g. `aten.as_strided_`)
crash during AOTAutograd's metadata collection phase. This affects any model whose
execution path triggers an inplace view op — `AdaptiveAvgPool1d/2d` is a common
example because ATen's implementation of adaptive pooling uses `aten.as_strided_`
internally.

## Error

```
RuntimeError: Check failed: xtensor: Input tensor is not an XLA tensor: XLAFloatType
```

## Root cause

The crash originates in `FunctionalTensorMode.__torch_dispatch__`
(`torch/_subclasses/functional_tensor.py:549-554`):

```python
# for metadata mutations, need to manually mutate the metadata of the FunctionalTensor wrapper
if (
    torch.Tag.inplace_view in func.tags
    and func is not torch.ops.aten.set_.source_Tensor
):
    with torch.utils._mode_utils.no_dispatch():
        func(*args, **kwargs)
```

After the main functionalization dispatch (line 511) has already converted the
inplace op to its functional equivalent, this block re-executes the inplace op
under `no_dispatch()` to **update the Python-side `FunctionalTensor` wrapper's
shape/stride metadata**.

The `args` are `FunctionalTensor` objects. These are wrapper subclass tensors
created via `_make_wrapper_subclass(..., device='xla:0', ...)` (at line 121-135
of the same file), inheriting the device from the inner tensor.

Under `no_dispatch()`, **ALL Python dispatch is disabled** — both
`TorchDispatchMode` instances (FunctionalTensorMode, FakeTensorMode) and tensor
subclass `__torch_dispatch__`. C++ dispatch takes over. It sees the XLA dispatch
key (derived from the wrapper's reported `device='xla:0'`) and routes
`aten.as_strided_` to XLA's kernel.

XLA's kernel calls `XLATensor::from_value()` expecting a native XLA tensor
object. Instead it receives a `_make_wrapper_subclass` wrapper that has no
underlying `XLATensor` storage. The check fails.

## Why it only affects XLA (and out-of-tree backends)

For CPU and CUDA (in-tree backends), `aten.as_strided_` under `no_dispatch()` on
a wrapper subclass is a **pure metadata mutation**: the kernel just updates the
tensor's shape/stride/storage_offset fields on the C++ `TensorImpl`. It doesn't
try to extract a backend-specific tensor type.

XLA (and likely other out-of-tree backends) register kernels for `as_strided_`
that expect their own tensor types. The wrapper subclass doesn't have such a
type, causing the crash.

## Why eager execution and naive GraphModule execution work

In eager mode (`model(xla_input)`) or when running the FX GraphModule directly
(`gm(xla_input)`), there is no `FunctionalTensor` wrapping and no
`no_dispatch()` call. XLA handles `aten.as_strided_` fine on its own native
tensor type. The problematic pattern is specifically the combination of:
`_make_wrapper_subclass(device='xla:0')` + `no_dispatch()` + XLA dispatch.

## How Inductor avoids this (decompositions are irrelevant)

Inductor never hits this bug because it only compiles for CPU and CUDA. For those
backends, `as_strided_` under `no_dispatch()` on wrapper subclasses works. This
has nothing to do with Inductor's decomposition tables — even though
`_adaptive_avg_pool2d` has a registered decomposition, there is no decomposition
for `adaptive_avg_pool1d`, and decompositions wouldn't help anyway because the
issue is with the `no_dispatch()` metadata fixup pattern, not with any specific
high-level op.

## Fakification timeline (who fakifies when)

1. **Dynamo** traces the model and may produce FakeTensor example_inputs
2. Real XLA tensors arrive at the backend (either directly from Dynamo or via defaking)
3. **AOTAutograd's `prepare_aot_module_simplified`** calls `construct_fake_mode()`
   then `process_inputs()` → `fake_mode.from_tensor(real_xla_tensor)` → creates
   new FakeTensors reporting `device='xla:0'`
4. **`run_functionalized_fw_and_collect_metadata`** wraps these FakeTensors in
   `FunctionalTensor` via `to_fun()` → `FunctionalTensor(FunctionalTensorWrapper(FakeTensor(xla:0)))`
5. The FX graph is executed. When an inplace_view op is encountered, the metadata
   fixup at line 553-554 triggers the crash.

## Workaround: AdaptiveAvgPool → torch.mean rewrite

Before AOTAutograd tracing, rewrite `call_module` nodes targeting
`AdaptiveAvgPool1d(1)` / `AdaptiveAvgPool2d((1,1))` to `torch.mean` with
appropriate `dim` and `keepdim=True`. This avoids `aten.as_strided_` entirely.

See `rewrite_adaptive_avgpool_to_mean` in `backend.py`.

This is a targeted workaround: any other op that triggers `aten.as_strided_`
(or another `inplace_view`-tagged op) during AOTAutograd tracing will hit the
same crash and require a similar rewrite.

## Failed approach: CPU-tracing hack

We attempted to move the GraphModule and example_inputs to CPU before passing
them to `aot_autograd`, so that `FunctionalTensor` wrappers would report
`device='cpu'` and the `no_dispatch()` metadata fixup would dispatch to CPU's
harmless metadata-only kernel. XLA tensors would be restored in `fw_compiler`.

This approach fails for three compounding reasons:

1. **Dynamo bakes device constants into FX graph nodes.** Operations like
   `.to(tensor.device)` or `torch.zeros(..., device=device)` get captured as
   literal `torch.device('xla', 0)` in node args and kwargs. Moving the
   GraphModule to CPU doesn't rewrite these constants, causing device mismatches
   at trace time. We wrote a graph rewriter to fix this, but the constants
   appear in args, kwargs, and as string `'xla:0'` values unpredictably.

2. **Non-parameter tensor attributes are not moved by `gm.cpu()`.** Tensors
   stored as plain attributes on submodules (accessed via `get_attr` nodes)
   remain on XLA. These must be discovered and moved individually.

3. **Symbolic shapes break across device moves.** Dynamo's symbolic shape
   environment (e.g. `s77`) is tied to the original XLA tensors. After moving
   to CPU, symbolic variables become disconnected, causing `KeyError` in
   `_tensorify_python_scalars`. This is fundamental and not fixable without
   changes to Dynamo's shape tracking.

## Suggested upstream fix

The metadata fixup at `functional_tensor.py:553-554` should not dispatch to
backend-specific kernels. Possible approaches:
1. Force dispatch to CPU/Meta backend for the metadata fixup call
2. Use a lower-level API to update shape/stride metadata without going through
   aten dispatch
3. Make the fixup operate on the wrapper's TensorImpl directly rather than
   re-executing the op
