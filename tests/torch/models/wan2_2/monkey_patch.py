# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0


# ---------------------------------------------------------------------------
# DiT monkey patches
# ---------------------------------------------------------------------------


def _patch_apply_lora_scale() -> None:
    """Make `@apply_lora_scale` a pass-through.

    `diffusers.utils.peft_utils.apply_lora_scale` wraps the DiT forward in
    a helper that calls `scale_lora_layers` + `unscale_lora_layers`, each
    of which is a graph break. This test loads plain weights via
    `load_dit()` – no LoRA adapters exist, so the wrapper is pure
    overhead.
    """
    from diffusers.utils import peft_utils

    def noop_decorator(kwargs_name: str = "joint_attention_kwargs"):
        def decorator(forward_fn):
            return forward_fn

        return decorator

    peft_utils.apply_lora_scale = noop_decorator

    # The WanTransformer3DModel.forward in diffusers is decorated at class
    # definition time, so the patch above only affects future imports.
    # Rebind the already-decorated forward to the underlying function.
    from diffusers.models.transformers.transformer_wan import WanTransformer3DModel

    wrapped = WanTransformer3DModel.forward
    underlying = getattr(wrapped, "__wrapped__", None)
    if underlying is not None:
        WanTransformer3DModel.forward = underlying


def _disable_tt_torch_function_override() -> None:
    """Pop `TorchFunctionOverride` off the global TorchFunctionMode stack.

    `tt_torch/torch_overrides.py` enters a `TorchFunctionMode` at import
    time. Its body is gated by `torch.compiler.is_compiling()` and does
    nothing on the compile path, but the mode still sits on dynamo's
    function-mode stack and forces a `__torch_function__` trace for every
    matmul / linear encountered during tracing.
    """
    try:
        import tt_torch.torch_overrides as overrides
    except ImportError:
        return

    mode = getattr(overrides, "torch_function_override", None)
    if mode is None:
        return

    try:
        mode.__exit__(None, None, None)
    except Exception:
        # Mode wasn't on the stack or was already popped – ignore.
        pass


def _patch_tt_torch_getitem_clamp() -> None:
    """Extend `TorchFunctionOverride` to silently clamp out-of-range slice
    endpoints on tensor `__getitem__`.

    CPU silently clamps slice `start` / `stop` values lying outside
    `[-size, size]`; torch-xla's lazy backend raises "Value out of range"
    instead. Upstream diffusers' `AutoencoderKLWan` (used by the Wan 2.2
    VAE decoder) relies on the CPU behavior — e.g. `x[:, :, -2:, :, :]`
    on a size-1 temporal dim. Intercept `Tensor.__getitem__` inside the
    existing `TorchFunctionMode` and rewrite the index in range before
    re-dispatching.
    """
    try:
        import tt_torch.torch_overrides as overrides
    except ImportError:
        return

    import torch

    def _clamp_oob_slice(tensor, idx):
        def clamp(s, size):
            start, stop, step = s.start, s.stop, s.step
            # Negative step uses different CPython bounds
            # (`max(-1, start + size)` vs. `max(0, start + size)`) that do
            # not round-trip through torch-xla's canonicalization — silent
            # wrong results would be worse than the raised error.
            if isinstance(step, int) and step < 0:
                return s
            changed = False
            if isinstance(start, int):
                if start < -size:
                    start, changed = -size, True
                elif start > size:
                    start, changed = size, True
            if isinstance(stop, int):
                if stop < -size:
                    stop, changed = -size, True
                elif stop > size:
                    stop, changed = size, True
            return slice(start, stop, s.step) if changed else s

        def dims_consumed(s):
            # Bool masks broadcast across `ndim` input dims and collapse
            # into one output dim; every other indexer covers exactly one.
            if isinstance(s, torch.Tensor) and s.dtype == torch.bool:
                return s.ndim
            return 1

        if isinstance(idx, slice):
            return clamp(idx, tensor.shape[0]) if tensor.ndim else idx

        if not isinstance(idx, tuple):
            return idx

        # Resolve `Ellipsis` span: dims the tuple explicitly addresses vs.
        # what's left for `...`. `None` / newaxis inserts an output dim
        # without consuming an input dim, so it's excluded from the count.
        explicit = sum(
            dims_consumed(s) for s in idx if s is not Ellipsis and s is not None
        )
        ellipsis_span = tensor.ndim - explicit

        out = []
        changed = False
        dim = 0
        for s in idx:
            if s is Ellipsis:
                out.append(s)
                dim += ellipsis_span
            elif s is None:
                out.append(s)
            elif isinstance(s, slice) and dim < tensor.ndim:
                clamped = clamp(s, tensor.shape[dim])
                out.append(clamped)
                changed = changed or clamped is not s
                dim += 1
            else:
                out.append(s)
                dim += dims_consumed(s)

        # Preserve the original `idx` object on no-op so the caller's
        # identity check can short-circuit re-dispatch.
        return tuple(out) if changed else idx

    original_torch_function = overrides.TorchFunctionOverride.__torch_function__

    def patched_torch_function(self, func, types, args, kwargs=None):
        if (
            func.__name__ == "__getitem__"
            and len(args) >= 2
            and isinstance(args[0], torch.Tensor)
        ):
            new_idx = _clamp_oob_slice(args[0], args[1])
            # Identity check: helper returns the original `args[1]` when
            # nothing needed clamping, so the common case pays one walk
            # and no re-dispatch.
            if new_idx is not args[1]:
                return func(args[0], new_idx, *args[2:], **(kwargs or {}))
        return original_torch_function(self, func, types, args, kwargs)

    overrides.TorchFunctionOverride.__torch_function__ = patched_torch_function
