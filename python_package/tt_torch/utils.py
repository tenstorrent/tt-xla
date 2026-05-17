# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Portions (c) 2026 Tenstorrent AI ULC

"""
Shared utilities for TT-Torch backend integration.
"""

import contextlib

import torch


class MockStream:
    """Mock stream class for TorchDynamo compilation with TT devices."""

    def __init__(self, device_index=0):
        self.device = torch.device("cpu", device_index)


def cpu_device_index():
    """Return CPU device index for TorchDynamo compatibility."""
    return 0


def cpu_stream(device_index=0):
    """Return mock CPU stream for TorchDynamo compatibility."""
    return MockStream(device_index)


def apply_dynamo_compatibility_patches():
    """Apply TorchDynamo compatibility patches for TT devices."""
    torch._C._accelerator_getDeviceIndex = cpu_device_index
    torch._C._accelerator_getStream = cpu_stream
    apply_dynamo_nn_module_wrap_values_patch()


# TODO: https://github.com/tenstorrent/tt-xla/issues/4806
_DYNAMO_WRAP_VALUES_BUG_TORCH_VERSION = "2.10.0"


def _wrap_values_patch_warn(msg: str) -> None:
    """
    Emit a warning about the ``wrap_values`` Dynamo patch.

    Prefers the project's ``ttxla_tools`` logger (matches the convention in
    the rest of ``tt_torch``), but lazily imported so this module stays
    importable even if ``ttxla_tools`` is not on the path yet. Falls back to
    the stdlib ``warnings`` module so the message is never silently dropped.
    """
    full_msg = f"[tt_torch] dynamo wrap_values patch: {msg}"
    try:
        from ttxla_tools.logging import logger

        logger.warning(full_msg)
    except Exception:
        import warnings

        warnings.warn(full_msg, RuntimeWarning, stacklevel=3)


def apply_dynamo_nn_module_wrap_values_patch() -> None:
    """
    Patch a bug in torch._dynamo's ``NNModuleVariable.call_method`` where the
    local helper ``wrap_values`` references an undefined ``named_children``
    free variable instead of its local ``result`` list. This raises
    ``InternalTorchDynamoError: NameError: cannot access free variable
    'named_children' ...`` whenever Dynamo traces ``nn.Module`` methods like
    ``.parameters()``, ``.children()``, ``.modules()`` or ``.buffers()`` (e.g.
    via transformers' ``invert_attention_mask`` -> ``self.dtype``).

    Introduced in pytorch/pytorch#167342, fixed upstream in
    pytorch/pytorch#174399. Only applied on the exact torch release that
    contains the bug (see ``_DYNAMO_WRAP_VALUES_BUG_TORCH_VERSION``); remove
    this patch once we move to a torch release containing the upstream fix.

    Safe to call multiple times (no-op after the first successful apply, or
    when running on a torch version that no longer contains the bug). Any
    unexpected failure inside this routine is surfaced as a warning rather
    than silently swallowed -- if the patch fails to apply on a torch
    version that *should* contain the bug, downstream Dynamo compilation
    errors would otherwise be hard to attribute.
    """
    # Strip any local-version suffix, e.g. "2.10.0+cpu" / "2.10.0+cu121".
    torch_version = torch.__version__.split("+", 1)[0]
    if torch_version != _DYNAMO_WRAP_VALUES_BUG_TORCH_VERSION:
        return

    try:
        from torch._dynamo.variables import nn_module as _nn_module_mod
        from torch._dynamo.variables.nn_module import NNModuleVariable
    except ImportError as exc:
        _wrap_values_patch_warn(
            f"could not import torch._dynamo.variables.nn_module ({exc!r}); "
            "patch not applied"
        )
        return

    if getattr(
        NNModuleVariable.call_method, "_tt_xla_wrap_values_patch_applied", False
    ):
        return

    import inspect
    import re
    import textwrap

    try:
        src = textwrap.dedent(inspect.getsource(NNModuleVariable.call_method))
    except (OSError, TypeError) as exc:
        _wrap_values_patch_warn(
            f"inspect.getsource(NNModuleVariable.call_method) failed "
            f"({exc!r}); patch not applied"
        )
        return

    # Anchor the rewrite on the `wrap_values` helper specifically rather than
    # on the bare ``named_children, mutation_type=...`` string. The helper is
    # the only place in ``call_method`` that mistakenly references
    # ``named_children`` (its local accumulator is called ``result``); every
    # other branch -- including the legitimate ``name == "named_children"``
    # branch and the ``name == "named_parameters"`` etc. branches -- builds
    # and returns a correctly-named list. Matching on ``def wrap_values(...
    # return ListIteratorVariable(named_children, ...)`` therefore stays
    # correct even if upstream later adds a third occurrence of the bare
    # marker above the buggy site.
    wrap_values_bug_re = re.compile(
        r"(def\s+wrap_values\s*\([^)]*\)[^:]*:"  # def wrap_values(items: ...) -> ...:
        r".*?"  # function body (non-greedy)
        r"return\s+ListIteratorVariable\(\s*)"  # return ListIteratorVariable(
        r"named_children"  # the misnamed free variable
        r"(\s*,\s*mutation_type\s*=\s*"  # , mutation_type=
        r"ValueMutationNew\(\)\s*\))",  # ValueMutationNew())
        flags=re.DOTALL,
    )
    match = wrap_values_bug_re.search(src)
    if match is None:
        _wrap_values_patch_warn(
            "could not locate the buggy `wrap_values` return in "
            f"NNModuleVariable.call_method source on torch {torch_version}; "
            "source layout likely changed, patch not applied"
        )
        return
    new_src = wrap_values_bug_re.sub(r"\1result\2", src, count=1)

    namespace: dict = {}
    try:
        exec(
            compile(new_src, _nn_module_mod.__file__, "exec"),
            _nn_module_mod.__dict__,
            namespace,
        )
    except Exception as exc:
        _wrap_values_patch_warn(
            f"failed to compile/exec rewritten call_method ({exc!r}); patch "
            "not applied"
        )
        return

    patched = namespace.get("call_method")
    if patched is None:
        _wrap_values_patch_warn(
            "rewritten module did not define `call_method`; patch not applied"
        )
        return
    patched._tt_xla_wrap_values_patch_applied = True
    NNModuleVariable.call_method = patched


@contextlib.contextmanager
def torch_dynamo_tt_device_compatibility():
    """
    Context manager to temporarily patch torch accelerator functions for TorchDynamo compilation with TT devices.

    This fixes TorchDynamo compilation errors when using TT devices by temporarily patching:
    - torch._C._accelerator_getDeviceIndex() to return CPU device index
    - torch._C._accelerator_getStream() to return CPU stream

    TorchDynamo calls these functions during compilation but they fail with TT devices,
    so we temporarily redirect them to CPU equivalents during compilation.
    """
    # Store original functions
    original_get_device_index = getattr(torch._C, "_accelerator_getDeviceIndex", None)
    original_get_stream = getattr(torch._C, "_accelerator_getStream", None)

    try:
        # Apply patches
        apply_dynamo_compatibility_patches()
        yield
    finally:
        # Restore original functions
        if original_get_device_index is not None:
            torch._C._accelerator_getDeviceIndex = original_get_device_index
        if original_get_stream is not None:
            torch._C._accelerator_getStream = original_get_stream


def apply_xla_dynamo_guard_repr_patch() -> None:
    """
    Apply patch to PyTorch Dynamo guard system to prevent XLA tensor materialization.

    This patches GuardBuilder.id_match_unchecked to avoid calling repr() on XLA tensors
    during guard construction, which would trigger expensive ReplicateShardedData operations.
    PyTorch uses repr(val) only to annotate verbose guard strings, so we replace it with
    a cheap metadata-only string for XLA tensors.

    The patch is safe to call multiple times (no-op after first application).
    Side effects: Modifies torch._dynamo.guards.GuardBuilder.id_match_unchecked globally.
    """
    try:
        from torch._dynamo.guards import GuardBuilder, get_verbose_code_parts
        from torch._dynamo.source import LocalSource, TypeSource
        from torch._guards import Guard
    except ImportError:
        # Silently skip if Dynamo modules are not available
        return

    # Check if patch is already applied
    if getattr(GuardBuilder.id_match_unchecked, "_tt_xla_guard_repr_patch", False):
        return

    def id_match_unchecked(self, guard, recompile_hint=None) -> None:
        # PyTorch uses repr(val) only to annotate verbose guard strings.
        # For XLA tensors/parameters, repr() calls tensor.to("cpu"), which
        # materializes sharded weights via ReplicateShardedData during guard
        # construction. Replace it with a cheap metadata-only string.
        if isinstance(guard.originating_source, TypeSource):
            return self.TYPE_MATCH(
                Guard(guard.originating_source.base, GuardBuilder.TYPE_MATCH)
            )

        ref = self.arg_ref(guard)
        val = self.get(guard)
        id_val = self.id_ref(val, guard.name)

        # Generate safe string representation without materializing XLA tensors
        try:
            if isinstance(val, torch.Tensor) and val.device.type == "xla":
                type_repr = f"<{type(val).__name__} device={val.device}>"
            else:
                type_repr = repr(val)
        except Exception:
            # Fallback for any repr() failures
            type_repr = f"<{type(val).__name__}>"

        code = f"___check_obj_id({ref}, {id_val}), type={type_repr}"
        self._set_guard_export_info(guard, [code], provided_func_name="ID_MATCH")
        self.get_guard_manager(guard).add_id_match_guard(
            id_val, get_verbose_code_parts(code, guard, recompile_hint)
        )

        # Handle LocalSource for Module objects
        if isinstance(guard.originating_source, LocalSource):
            if isinstance(val, torch.nn.Module):
                local_name = guard.originating_source.local_name
                weak_id = self.lookup_weakrefs(val)
                if weak_id is not None:
                    self.id_matched_objs[local_name] = weak_id

    # Mark the function as patched and apply the patch
    id_match_unchecked._tt_xla_guard_repr_patch = True
    GuardBuilder.id_match_unchecked = id_match_unchecked
