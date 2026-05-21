# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
tt-lang integration for tt-xla.

Author-facing surface: the ``@tt_torch.kernel`` decorator. Wrap a tt-lang
kernel (typically also decorated with ``@ttl.operation``) with a stable
``kernel_id`` and the decorator turns invocations into
``stablehlo.custom_call @tt.tt_lang_op`` operations that flow through the
normal SHLO -> TTIR -> TTNN pipeline. The torch custom op itself lives in
:mod:`tt_torch.custom_ops` alongside the rest of the ``tt::*`` ops.

Pipeline at a glance::

    @tt_torch.kernel(kernel_id="...")        registers callable, emits custom op
                |
                v
    stablehlo.custom_call @tt.tt_lang_op { kernel_id, arg_roles, version_tag }
                |  (opaque through SHLO / Shardy frontend)
                v
    StableHLO -> TTIR  (ttir.tt_lang_op, kernel_id / arg_roles preserved)
                |
                v
    TTIR -> TTNN       (ttnn.tt_lang_op, kernel_artifact empty)
                |
                v
    ModuleBuilder::resolveTtLangKernels (pjrt_plugin_tt)
                |  embedded Pybind11 bridge calls resolve_kernel(...)
                |  populates the kernel_artifact attribute in place
                v
    TTNN -> Flatbuffer  (lowered to ttnn.generic with the compiled kernel
                         body embedded; "out"-roled operand TensorRefs
                         aliased to results -- see TTNNToFlatbuffer.cpp)
                |
                v
    PJRT executable returned to torch-xla
                |
                v
    runtime executes ttnn.generic on device

The plugin -> Python callback lives in
``pjrt_implementation/src/api/module_builder/tt_lang_bridge.cc`` and
invokes :func:`resolve_kernel` below. The end-to-end path is exercised by
``tests/torch/ops/test_tt_lang_kernel_e2e.py`` against real hardware.

The Python side of the integration has three responsibilities:

1. A registry mapping ``kernel_id`` to the user's callable.
2. The decorator that emits the custom op and populates the registry.
3. A ``resolve_kernel`` entry point with a stable signature that the
   plugin bridge invokes during ``ModuleBuilder::resolveTtLangKernels``.

Caching, ABI ctypes mirroring, MLIR-dump capture, and backend selection
were intentionally cut. None of them have been needed against the real
plugin or real tt-lang yet; each can come back as a small targeted change
the day it's needed. See ``docs/src/tt_lang_integration.md`` for the
forward design and the open work items.
"""

from __future__ import annotations

import functools
import hashlib
import inspect
import threading
from typing import Any, Callable, List, Optional, Sequence, Tuple

import torch

from tt_torch.custom_ops import tt_lang_op_dispatch


__all__ = [
    "KernelEntry",
    "TtLangError",
    "get_registered_kernel",
    "iter_registered_kernels",
    "kernel",
    "resolve_kernel",
]


_VALID_ROLES = frozenset({"in", "out"})


class TtLangError(RuntimeError):
    """Raised when the tt-lang integration is asked to do something it can't."""


class KernelEntry:
    """A registered tt-lang kernel.

    Only the fields that the decorator, the CPU fallback, and the future
    resolve callback actually use. Add fields back only when something
    concrete needs them.
    """

    __slots__ = ("kernel_id", "impl", "version_tag", "arg_roles")

    def __init__(
        self,
        *,
        kernel_id: str,
        impl: Callable,
        version_tag: str,
        arg_roles: Tuple[str, ...],
    ) -> None:
        self.kernel_id = kernel_id
        self.impl = impl
        self.version_tag = version_tag
        self.arg_roles = arg_roles

    def __repr__(self) -> str:
        return (
            f"KernelEntry(kernel_id={self.kernel_id!r}, "
            f"version_tag={self.version_tag!r}, arg_roles={self.arg_roles!r})"
        )


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------


_REGISTRY: dict[str, KernelEntry] = {}
_REGISTRY_LOCK = threading.Lock()


def get_registered_kernel(kernel_id: str) -> KernelEntry:
    """Look up a kernel by ``kernel_id``. Raises ``KeyError`` if missing."""
    with _REGISTRY_LOCK:
        entry = _REGISTRY.get(kernel_id)
    if entry is None:
        raise KeyError(f"No tt-lang kernel registered with id {kernel_id!r}.")
    return entry


def iter_registered_kernels() -> List[KernelEntry]:
    """Snapshot of all currently-registered kernels (registration order)."""
    with _REGISTRY_LOCK:
        return list(_REGISTRY.values())


def _register(entry: KernelEntry) -> None:
    with _REGISTRY_LOCK:
        prev = _REGISTRY.get(entry.kernel_id)
        if prev is not None and prev.version_tag != entry.version_tag:
            raise ValueError(
                f"tt-lang kernel_id {entry.kernel_id!r} is already registered "
                f"with a different version_tag "
                f"(existing={prev.version_tag!r}, new={entry.version_tag!r}). "
                f"Bump kernel_id when the kernel changes."
            )
        _REGISTRY[entry.kernel_id] = entry


def _clear_registry_for_tests() -> None:
    """Test helper. Not part of the public API."""
    with _REGISTRY_LOCK:
        _REGISTRY.clear()


def _derive_version_tag(fn: Callable) -> str:
    """Stable short hash of the kernel source. Best-effort.

    Falls back to the qualified name if the source isn't introspectable
    (e.g. C-backed callables); identical source produces identical tags.
    """
    try:
        material = inspect.getsource(fn)
    except (OSError, TypeError):
        material = getattr(fn, "__qualname__", fn.__class__.__name__)
    return hashlib.sha1(material.encode("utf-8")).hexdigest()[:12]


# ---------------------------------------------------------------------------
# Argument-role inference
# ---------------------------------------------------------------------------


def _normalize_arg_roles(
    fn: Callable, roles: Optional[Sequence[str]], num_args: int
) -> Tuple[str, ...]:
    """Validate ``arg_roles`` into a tuple of ``"in"``/``"out"`` tokens.

    Raises ``ValueError`` if ``roles`` is ``None`` (every kernel must
    declare its roles explicitly -- there is no name-based inference),
    if ``len(roles) != num_args``, or if any token is not in
    ``_VALID_ROLES``.
    """
    if roles is None:
        raise ValueError(
            "tt-lang kernels must declare arg_roles= explicitly; name-based "
            "inference was removed. Pass e.g. arg_roles=(\"in\", \"in\", \"out\")."
        )
    norm = tuple(roles)
    if len(norm) != num_args:
        raise ValueError(
            f"arg_roles has {len(norm)} entries but kernel was called with "
            f"{num_args} positional tensors."
        )
    bad = [r for r in norm if r not in _VALID_ROLES]
    if bad:
        raise ValueError(
            f"arg_roles must contain only {sorted(_VALID_ROLES)!r}; got {bad!r}."
        )
    return norm


def _positional_arg_count(fn: Callable) -> int:
    """Return the static positional-arg count of ``fn``, or ``0`` if unknown.

    Counts ``POSITIONAL_ONLY`` and ``POSITIONAL_OR_KEYWORD`` parameters.
    Returns ``0`` as a sentinel for ``*args`` kernels (arity unknown
    until call time) and for callables whose signature can't be
    introspected -- the decoration-time length check is then skipped
    and the real validation happens at call time against ``len(tensors)``.
    """
    try:
        sig = inspect.signature(fn)
    except (TypeError, ValueError):
        return 0
    count = 0
    for p in sig.parameters.values():
        if p.kind in (
            inspect.Parameter.POSITIONAL_ONLY,
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
        ):
            count += 1
        elif p.kind == inspect.Parameter.VAR_POSITIONAL:
            return 0
    return count


# ---------------------------------------------------------------------------
# Decorator
# ---------------------------------------------------------------------------


def kernel(
    *,
    kernel_id: str,
    arg_roles: Sequence[str],
    shard_spec: str = "",
    version_tag: Optional[str] = None,
) -> Callable[[Callable], Callable]:
    """Register a tt-lang kernel and emit a ``stablehlo.custom_call`` on XLA.

    Parameters
    ----------
    kernel_id:
        Stable identifier carried through the StableHLO program. Used by
        the plugin's eventual resolve callback to look up the live kernel.
        Must be unique within the process; whitespace is disallowed so the
        id round-trips cleanly through ``frontend_attributes``.
    arg_roles:
        Required sequence of ``"in"`` / ``"out"`` tags, one per positional
        tensor argument. Free-form ordering (e.g. ``("in", "out", "in")``)
        is supported and preserved end-to-end through MLIR -- see the
        ``MemoryEffectOpInterface`` discussion in
        ``docs/src/tt_lang_integration.md``.
    shard_spec:
        Optional opaque sharding hint, passed through verbatim as a
        ``frontend_attribute``.
    version_tag:
        Cache-busting tag. Defaults to a short hash of the kernel source.

    Returns
    -------
    Callable
        The wrapped kernel. The return mirrors the ``"out"`` arguments: a
        single tensor when exactly one, a tuple in declaration order
        otherwise. The wrapped kernel requires XLA tensors -- calls with
        non-XLA tensors raise ``NotImplementedError``. There is no CPU
        fallback by design: a "ran on CPU and got a plausible-looking
        answer" outcome is indistinguishable from "the hardware kernel
        is silently broken," so the wrapper refuses to compute outside
        XLA at all.
    """
    if not isinstance(kernel_id, str) or not kernel_id:
        raise ValueError("kernel_id must be a non-empty string.")
    if any(c.isspace() for c in kernel_id):
        raise ValueError("kernel_id must not contain whitespace.")

    def decorator(fn: Callable) -> Callable:
        vt = version_tag or _derive_version_tag(fn)
        entry = KernelEntry(
            kernel_id=kernel_id,
            impl=fn,
            version_tag=vt,
            arg_roles=_normalize_arg_roles(fn, arg_roles, _positional_arg_count(fn)),
        )
        _register(entry)

        @functools.wraps(fn)
        def wrapper(*tensors: torch.Tensor):
            if not tensors:
                raise TypeError(
                    f"tt-lang kernel {kernel_id!r} was called with no tensors."
                )
            for i, t in enumerate(tensors):
                if not isinstance(t, torch.Tensor):
                    raise TypeError(
                        f"tt-lang kernel {kernel_id!r}: positional arg {i} is "
                        f"not a torch.Tensor (got {type(t).__name__})."
                    )

            roles = _normalize_arg_roles(fn, arg_roles, len(tensors))
            out_indices = [i for i, r in enumerate(roles) if r == "out"]
            if not out_indices:
                raise ValueError(
                    f"tt-lang kernel {kernel_id!r}: at least one argument must "
                    f"be tagged 'out' (mutation-style outputs are required)."
                )

            device = tensors[0].device
            for t in tensors[1:]:
                if t.device != device:
                    raise ValueError(
                        f"tt-lang kernel {kernel_id!r}: all tensors must share "
                        f"a device (saw {device} and {t.device})."
                    )

            if device.type != "xla":
                # No CPU fallback by design -- see the decorator docstring.
                raise NotImplementedError(
                    f"tt-lang kernel {kernel_id!r} can only run on XLA "
                    f"tensors (saw device={device}). Move inputs onto the "
                    f"XLA device with .to(xm.xla_device())."
                )

            outputs = tt_lang_op_dispatch(
                list(tensors),
                kernel_id=kernel_id,
                arg_roles=",".join(roles),
                version_tag=vt,
                shard_spec=shard_spec,
                out_indices=out_indices,
            )
            # Mutation-style API: write the functional results back into
            # the caller's pre-allocated outputs.
            for idx, result in zip(out_indices, outputs):
                if tensors[idx] is not result:
                    tensors[idx].copy_(result)
            return outputs[0] if len(outputs) == 1 else tuple(outputs)

        wrapper._tt_lang_kernel_entry = entry  # type: ignore[attr-defined]
        return wrapper

    return decorator


# ---------------------------------------------------------------------------
# Resolve entry point (called by the plugin's tt_lang_bridge)
# ---------------------------------------------------------------------------


def resolve_kernel(
    *,
    kernel_id: str,
    version_tag: str,
    shapes: Sequence[Sequence[int]],
    dtypes: Sequence[str],
    layouts: Sequence[str] = (),
    mesh_shape: Sequence[int] = (1,),
    options: Optional[dict] = None,
    arg_roles: Optional[str] = None,
    shard_spec: Optional[str] = None,
    **_unused: Any,
) -> bytes:
    """Resolve a registered kernel to a compiled artifact.

    Stub entry point. The signature mirrors what the plugin's
    ``pjrt_plugin_tt::tt_lang_bridge::resolveKernels`` will call (under
    the GIL, via pybind11) once the cross-stack pieces land. The body
    is intentionally a ``NotImplementedError`` until then -- the
    decorator surface plus the registry are useful on their own (they
    cover the entire CPU-fallback path).

    Subsequent commits in this branch fill in the compile driver, the
    stub-tensor compile path, and the JSON serializer.

    The function does NOT take ``self`` / ``cls`` and uses keyword-only
    arguments so the bridge can pass extra metadata forward without
    breaking the call signature.
    """
    # Validate the registry lookup eagerly so a stale call site fails
    # with a clear message rather than NotImplementedError.
    entry = get_registered_kernel(kernel_id)
    if entry.version_tag != version_tag:
        raise TtLangError(
            f"tt-lang kernel {kernel_id!r}: registered version_tag "
            f"{entry.version_tag!r} does not match callback's "
            f"{version_tag!r}. The kernel source was edited after the "
            f"StableHLO program was cached -- re-run the calling script."
        )
    raise NotImplementedError(
        "tt_torch.tt_lang.resolve_kernel is not yet implemented. The "
        "compile driver lands in a follow-up commit on this branch."
    )
