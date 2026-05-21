# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""tt-lang integration for tt-xla.

Author-facing surface: the ``@tt_torch.kernel`` decorator. Wrap a
tt-lang kernel (typically also decorated with ``@ttl.operation``) with
a stable ``kernel_id`` and invocations on XLA tensors become
``stablehlo.custom_call @tt.tt_lang_op`` ops that flow through the
SHLO -> TTIR -> TTNN pipeline. The plugin's
``ModuleBuilder::resolveTtLangKernels`` calls back into
:func:`resolve_kernel` here to populate each op's ``kernel_artifact``.

The full pipeline and design notes live in
``docs/source/tt_lang_integration.md``. The torch custom op definition
lives in :mod:`tt_torch.custom_ops` alongside the rest of ``tt::*``.
"""

from __future__ import annotations

import functools
import hashlib
import inspect
import json
import os
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


# Format version of the JSON artifact returned by ``resolve_kernel``.
# The tt-mlir flatbuffer emitter checks the same value; bump on any
# breaking schema change. Schema documented at the
# ``_serialize_compiled_kernel`` definition below.
_ARTIFACT_FORMAT_VERSION = 1


_VALID_ROLES = frozenset({"in", "out"})


class TtLangError(RuntimeError):
    """Raised when the tt-lang integration is asked to do something it can't."""


class KernelEntry:
    """A registered tt-lang kernel."""

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
            'inference was removed. Pass e.g. arg_roles=("in", "in", "out").'
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
        ``docs/source/tt_lang_integration.md``.
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
# tt-lang compile driver
# ---------------------------------------------------------------------------
#
# Invoked from the plugin's tt_lang_bridge once per ``ttnn.tt_lang_op``:
# look up the registered ``KernelEntry``, drive tt-lang's compile path
# against device-less stubs with ``TTLANG_COMPILE_ONLY=1`` (capturing the
# ``CompiledTTNNKernel`` via a monkey-patched ``_compile_kernel``), then
# serialize the result into the JSON the tt-mlir flatbuffer emitter
# consumes. Reaching into the private ``_compile_kernel`` is the DEMO
# HACK described above; it goes away once tt-lang exposes a stable
# spec-based entry point.

# MLIR element-type spelling -> torch.dtype. The plugin sends these as
# the printed form of each operand's element type. Unsigned widths fold
# onto signed equivalents because the stub tensors are never read by
# the kernel itself; they only satisfy tt-lang's ``register_tensor_name``
# / ``_track_tensor_sources`` validation.
_MLIR_DTYPE_TO_TORCH: dict[str, torch.dtype] = {
    "f64": torch.float64,
    "f32": torch.float32,
    "f16": torch.float16,
    "bf16": torch.bfloat16,
    "i64": torch.int64,
    "i32": torch.int32,
    "i16": torch.int16,
    "i8": torch.int8,
    "i1": torch.bool,
    "bool": torch.bool,
    "ui64": torch.int64,
    "ui32": torch.int32,
    "ui16": torch.int16,
    "ui8": torch.uint8,
}


def _torch_dtype_from_mlir_string(name: str) -> torch.dtype:
    """Map an MLIR-style element-type string to a ``torch.dtype``.

    Unknown / non-element types fall back to ``torch.float32`` so the
    compile can still proceed (tt-lang doesn't read dtype off the stub
    tensor on the compile-only path).
    """
    return _MLIR_DTYPE_TO_TORCH.get(name.strip().lower(), torch.float32)


# ---------------------------------------------------------------------------
# Device-less compile stand-ins (DEMO HACK)
# ---------------------------------------------------------------------------
#
# tt-lang's compile-only path needs ``(shape, dtype, layout,
# memory_space, grid)`` metadata, but its entry point insists on real
# ``ttnn.Tensor`` arguments and runs ``isinstance(arg, ttnn.Tensor)``
# checks. Opening a real ttnn device in the plugin process races against
# PJRT's own ``init_command_queue_device_with_topology`` (both register
# kernel binaries in their own ``Program`` caches), so we feed tt-lang
# duck-typed stand-ins and monkey-patch ``is_ttnn_tensor`` for the
# duration of the compile. See ``docs/source/tt_lang_integration.md``
# ("Device-less compile path") for the full discussion; the proper fix
# is a ``_compile_ttnn_kernel_from_spec`` entry point upstream in
# tt-lang, after which these stubs and the monkey-patch can be deleted.
# Grep for ``DEMO HACK`` to find every site.


class _StubGridSize:
    """Return value of ``device.compute_with_storage_grid_size()``."""

    __slots__ = ("x", "y")

    def __init__(self, x: int = 8, y: int = 8) -> None:
        self.x = x
        self.y = y


class _StubTtnnDevice:
    """Stand-in for ``ttnn.Device`` (compile path).

    Exposes ``compute_with_storage_grid_size()`` (read by tt-lang's
    grid resolution) and ``arch``. Intentionally omits ``.shape`` so
    tt-lang's ``_is_mesh_tensor`` gate returns False.
    """

    __slots__ = ("_grid", "arch")

    def __init__(
        self,
        *,
        grid_cols: int = 8,
        grid_rows: int = 8,
        arch: str = "wormhole_b0",
    ) -> None:
        self._grid = _StubGridSize(grid_cols, grid_rows)
        self.arch = arch

    def compute_with_storage_grid_size(self) -> _StubGridSize:
        return self._grid


class _StubMemoryConfig:
    """Return value of ``tensor.memory_config()``.

    tt-lang reads ``buffer_type`` and ``str()``s it; the substring scan
    in ``_detect_memory_space_from_tensor`` classifies that as L1/DRAM.
    """

    __slots__ = ("buffer_type",)

    def __init__(self, buffer_type: str = "DRAM") -> None:
        self.buffer_type = buffer_type


class _StubTtnnTensor:
    """Stand-in for ``ttnn.Tensor`` on the compile-only path.

    Exposes exactly what tt-lang inspects during ``_compile_ttnn_kernel``:
    ``shape``, ``dtype``, ``layout`` (must contain ``"TILE"`` to pass
    the tilized-only gate), ``memory_config()``, and ``device()``.
    ``isinstance(stub, ttnn.Tensor)`` is False, so ``_drive_ttl_compile``
    additionally monkey-patches ``ttl.*.is_ttnn_tensor`` to accept it.
    """

    __slots__ = ("_shape", "dtype", "layout", "_memory_config", "_device")

    def __init__(
        self,
        *,
        shape: Sequence[int],
        dtype: Any,
        layout: str,
        memory_config: _StubMemoryConfig,
        device: _StubTtnnDevice,
    ) -> None:
        self._shape = tuple(int(s) for s in shape)
        self.dtype = dtype
        self.layout = layout
        self._memory_config = memory_config
        self._device = device

    @property
    def shape(self) -> Tuple[int, ...]:
        return self._shape

    def memory_config(self) -> _StubMemoryConfig:
        return self._memory_config

    def device(self) -> _StubTtnnDevice:
        return self._device

    def __repr__(self) -> str:
        return (
            f"_StubTtnnTensor(shape={self._shape!r}, dtype={self.dtype!r}, "
            f"layout={self.layout!r}, mem={self._memory_config.buffer_type!r})"
        )


def _memory_space_from_layout(layout_str: str) -> str:
    """Map a printed ``#ttnn.ttnn_layout<...>`` substring to L1/DRAM.

    Returns the bare string the stub's ``buffer_type`` should contain.
    The default is ``"DRAM"`` because that's what tt-lang itself
    defaults to when no explicit memory_config is plumbed through.
    """
    s = layout_str.lower()
    if "#l1" in s or " l1>" in s or "l1_memory" in s:
        return "L1"
    return "DRAM"


def _make_deviceless_ttnn_args(
    shapes: Sequence[Sequence[int]],
    dtypes: Sequence[str],
    layouts: Sequence[str],
) -> List[_StubTtnnTensor]:
    """Build a list of ``_StubTtnnTensor`` stand-ins matching the
    bridge's per-operand metadata.

    The shared ``_StubTtnnDevice`` instance is what
    ``Operation.__call__`` and ``_resolve_grid`` see when they call
    ``_require_device`` on the args. We hand back an 8x8 Wormhole_b0
    grid because that's the device family this integration targets;
    architecture- or board-conditional grids are a follow-up once
    upstream gives us a real TensorSpec entry point.
    """
    if not layouts:
        layouts = [""] * len(shapes)
    if len(shapes) != len(dtypes) or len(shapes) != len(layouts):
        raise TtLangError(
            f"resolve_kernel: shapes/dtypes/layouts length mismatch "
            f"({len(shapes)}/{len(dtypes)}/{len(layouts)})."
        )
    device = _StubTtnnDevice()
    args: List[_StubTtnnTensor] = []
    for shape, dtype_str, layout_str in zip(shapes, dtypes, layouts):
        mem_cfg = _StubMemoryConfig(_memory_space_from_layout(layout_str or ""))
        args.append(
            _StubTtnnTensor(
                shape=shape,
                dtype=_torch_dtype_from_mlir_string(dtype_str),
                layout="TILE_LAYOUT",
                memory_config=mem_cfg,
                device=device,
            )
        )
    return args


_DRIVE_LOCK = threading.Lock()


def _drive_ttl_compile(entry: "KernelEntry", mock_args: Sequence[Any]):
    """Run tt-lang's compile path for ``entry.impl`` and capture the
    resulting ``CompiledTTNNKernel`` without executing it.

    Single-flighted under ``_DRIVE_LOCK`` because we monkey-patch a
    module-global for the duration of the call. The plugin holds the GIL
    while invoking us, so contention here is theoretical, but the lock
    keeps the patch + env-var dance robust against any future caller
    that decides to thread.
    """
    try:
        import ttl.ttl_api as _ttl_api  # type: ignore
    except ImportError as e:
        raise TtLangError(
            "resolve_kernel: tt-lang is not importable in the current "
            "Python environment. Install the tt-lang wheel into the same "
            "venv that loaded the pjrt_plugin_tt plugin so the compile "
            f"path is available. (Original error: {e})"
        )

    captured: List[Any] = []
    original_compile = _ttl_api._compile_kernel

    @functools.wraps(original_compile)
    def _capturing_compile(*args, **kwargs):
        compiled = original_compile(*args, **kwargs)
        captured.append(compiled)
        return compiled

    # DEMO HACK: ``is_ttnn_tensor`` is re-exported from
    # ``ttl.dtype_utils`` into multiple modules via ``from ... import``,
    # so each rebind has its own copy. Patch every site that already
    # defines the attribute so our ``_StubTtnnTensor`` instances pass
    # tt-lang's gate; restore on exit. New rebind sites added upstream
    # surface as ``"No device found: no ttnn tensor arguments were
    # provided"`` from ``_require_device`` -- add the module here.
    _PATCH_MODULES = ("ttl.dtype_utils", "ttl.ttl_api", "ttl._src.ttl_ast")
    using_stubs = any(isinstance(a, _StubTtnnTensor) for a in mock_args)
    patch_sites: List[Tuple[Any, str, Any, bool]] = []
    if using_stubs:
        for module_name in _PATCH_MODULES:
            try:
                module = __import__(module_name, fromlist=["is_ttnn_tensor"])
            except ImportError:
                continue
            had_attr = hasattr(module, "is_ttnn_tensor")
            original = getattr(module, "is_ttnn_tensor", None)

            def _accept_stub(t: Any, _orig=original) -> bool:
                if isinstance(t, _StubTtnnTensor):
                    return True
                return bool(_orig(t)) if _orig is not None else False

            patch_sites.append((module, "is_ttnn_tensor", original, had_attr))
            setattr(module, "is_ttnn_tensor", _accept_stub)

    prev_env = os.environ.get("TTLANG_COMPILE_ONLY")
    with _DRIVE_LOCK:
        _ttl_api._compile_kernel = _capturing_compile  # type: ignore[attr-defined]
        os.environ["TTLANG_COMPILE_ONLY"] = "1"
        try:
            entry.impl(*mock_args)
        finally:
            _ttl_api._compile_kernel = original_compile  # type: ignore[attr-defined]
            for module, attr, original, had_attr in patch_sites:
                if had_attr:
                    setattr(module, attr, original)
                else:
                    try:
                        delattr(module, attr)
                    except AttributeError:
                        pass
            if prev_env is None:
                os.environ.pop("TTLANG_COMPILE_ONLY", None)
            else:
                os.environ["TTLANG_COMPILE_ONLY"] = prev_env

    if not captured or captured[0] is None:
        raise TtLangError(
            f"tt-lang compile did not produce a CompiledTTNNKernel for "
            f"kernel_id={entry.kernel_id!r}. The kernel may have hit a "
            f"compile error (check stderr for tt-lang diagnostics)."
        )
    # `entry.impl` may end up triggering multiple compiles (e.g. nested
    # @ttl.operation calls in unusual kernels); we take the last one,
    # which is the outermost-finishing compile.
    return captured[-1]


# Mapping from a tt-metal/ttnn DataType (or torch dtype) to the ttcore
# DataType enum name that the tt-mlir flatbuffer emitter expects. Both
# producers (this file) and consumer (TTNNToFlatbuffer.cpp's
# `parseDataType`) must stay in sync with
# `third_party/tt-mlir/.../Target/Common/types.fbs::enum DataType`.
_TTNN_DTYPE_NAME_TO_FB: dict[str, str] = {
    "BFLOAT16": "BFloat16",
    "FLOAT32": "Float32",
    # tt-metal hardware implements f16 as bf16; tt-lang's
    # `format_name_to_ttnn_dtype("float16")` also returns BFLOAT16,
    # so this entry exists only to handle the very rare ttnn enum that
    # spells the value as `FLOAT16`.
    "FLOAT16": "BFloat16",
    "BFLOAT8_B": "BFP_BFloat8",
    "BFLOAT4_B": "BFP_BFloat4",
    "INT32": "Int32",
    "UINT32": "UInt32",
    "UINT16": "UInt16",
    "UINT8": "UInt8",
}

_TORCH_DTYPE_TO_FB: dict[torch.dtype, str] = {
    torch.bfloat16: "BFloat16",
    torch.float32: "Float32",
    torch.float16: "BFloat16",
    torch.int32: "Int32",
    torch.uint8: "UInt8",
    torch.bool: "UInt8",
}


def _ttnn_dtype_name(dtype: Any) -> str:
    """Best-effort ``data_format`` name lookup (matches ttnn.DataType.name)."""
    # ttnn enum exposes `.name`; torch.dtype does not.
    name = getattr(dtype, "name", None)
    if isinstance(name, str):
        return name.upper()
    return str(dtype).rsplit(".", 1)[-1].upper()


def _dtype_to_flatbuffer_name(dtype: Any) -> str:
    """Map an arbitrary tensor dtype to the ttcore DataType enum name.

    Accepts ``ttnn.DataType`` enums, ``torch.dtype`` values, and string
    forms (e.g. ``"bfloat16"``). Raises ``TtLangError`` if the value
    can't be mapped, because emitting a wrong enum would produce a
    silently-broken flatbuffer.
    """
    if isinstance(dtype, torch.dtype):
        try:
            return _TORCH_DTYPE_TO_FB[dtype]
        except KeyError as e:
            raise TtLangError(
                f"resolve_kernel: cannot map torch dtype {dtype!r} to a "
                f"ttcore DataType."
            ) from e

    name = _ttnn_dtype_name(dtype)
    try:
        return _TTNN_DTYPE_NAME_TO_FB[name]
    except KeyError as e:
        raise TtLangError(
            f"resolve_kernel: cannot map ttnn DataType {name!r} to a "
            f"ttcore DataType."
        ) from e


def _tile_bytes_from_dtype_name(name: str) -> int:
    """Tile byte size for a ttcore DataType name.

    Mirrors ``tt-lang/dtype_utils::tile_bytes_from_dtype`` but indexed by
    flatbuffer enum name so we don't have to import ttnn just to compute
    sizes. Each TT-Metal tile is 32x32 elements; the byte cost depends
    on element width and format-specific exponent overhead.
    """
    return {
        "BFloat16": 32 * 32 * 2,
        "Float32": 32 * 32 * 4,
        "Int32": 32 * 32 * 4,
        "UInt32": 32 * 32 * 4,
        "UInt16": 32 * 32 * 2,
        "UInt8": 32 * 32,
        "BFP_BFloat8": 32 * 32 + 64,
        "BFP_BFloat4": 512 + 64,
    }.get(name, 32 * 32 * 2)


def _serialize_kernel_config(thread_type: str, cfg: Any, noc_kernel_idx: int) -> dict:
    """Turn a tt-lang kernel-config descriptor into a structured JSON dict.

    The flatbuffer ``KernelConfig`` union has four members; tt-lang
    emits ``ComputeKernelConfig`` for compute threads and a Reader/Writer
    pair for NOC threads (mapped by metal onto NCRISC+Noc1 / BRISC+Noc0,
    respectively). Fields are surfaced by name so the C++ emitter does
    not have to mirror tt-lang's Python descriptor classes.
    """
    if thread_type == "compute":
        return {
            "type": "ComputeKernelConfig",
            "math_fidelity": str(getattr(cfg, "math_fidelity", "HiFi4")).split(".")[-1],
            "fp32_dest_acc_en": bool(getattr(cfg, "fp32_dest_acc_en", False)),
            "dst_full_sync_en": bool(getattr(cfg, "dst_full_sync_en", False)),
            "bfp8_pack_precise": bool(getattr(cfg, "bfp8_pack_precise", False)),
            "math_approx_mode": bool(getattr(cfg, "math_approx_mode", False)),
        }
    if thread_type == "noc":
        # tt-lang assigns the first noc kernel to NCRISC (reader) and the
        # second to BRISC (writer); see `_compile_ttnn_kernel`.
        if noc_kernel_idx == 0:
            return {"type": "ReaderKernelConfig"}
        return {"type": "WriterKernelConfig"}
    raise TtLangError(
        f"resolve_kernel: unknown kernel thread_type {thread_type!r}; "
        f"expected 'compute' or 'noc'."
    )


def _serialize_core_range(core_ranges: Any) -> dict:
    """Extract a single ``CoreRange`` rectangle from a ``ttnn.CoreRangeSet``.

    tt-lang only emits a single bounding-box rectangle today, so the
    artifact schema models ``core_range`` as one ``{start, end}`` pair.
    Multi-rectangle support is a future schema bump.
    """
    bbox = core_ranges.bounding_box()
    start = bbox.start_coord if hasattr(bbox, "start_coord") else bbox.start
    end = bbox.end_coord if hasattr(bbox, "end_coord") else bbox.end
    return {
        "start": [int(start.x), int(start.y)],
        "end": [int(end.x), int(end.y)],
    }


def _serialize_cb_config(cb: Any) -> dict:
    """Turn a tt-lang DFB / ``CompilerAllocatedDFBConfig`` into a structured
    CB descriptor.

    Mirrors the same byte-size math as
    ``tt-lang/kernel_runner.py::build_cb_descriptors`` so that the
    flatbuffer's ``KernelCBDescriptor`` ends up byte-for-byte identical to
    what tt-lang would have constructed at native launch time. Anything
    we don't recognise raises -- silent truncation here turns into a
    runtime fault later.
    """
    # CompilerAllocatedDFBConfig is a dataclass with explicit fields.
    if (
        hasattr(cb, "data_format")
        and hasattr(cb, "num_tiles")
        and hasattr(cb, "block_count")
        and not hasattr(cb, "tensor")
    ):
        # Compiler-allocated DFB: format comes from a name string.
        name_map = {"bfloat16": "BFloat16", "float16": "BFloat16", "float32": "Float32"}
        data_format = name_map.get(cb.data_format.lower(), "BFloat16")
        page_size = _tile_bytes_from_dtype_name(data_format)
        total_size = int(cb.num_tiles) * int(cb.block_count) * page_size
        return {
            "buffer_index": int(getattr(cb, "_cb_index", -1)),
            "data_format": data_format,
            "page_size": page_size,
            "total_size": total_size,
            "num_tiles": int(cb.num_tiles),
            "block_count": int(cb.block_count),
        }

    # User-declared DataflowBuffer.
    ref = cb.tensor
    data_format = _dtype_to_flatbuffer_name(ref.dtype)
    page_size = _tile_bytes_from_dtype_name(data_format)
    shape = tuple(int(s) for s in cb.shape)
    if len(shape) < 2:
        raise TtLangError(
            f"resolve_kernel: DFB shape {shape!r} must have at least 2 dims."
        )
    num_tiles = shape[0] * shape[1] * int(cb.block_count)
    return {
        "buffer_index": int(getattr(cb, "_cb_index", -1)),
        "data_format": data_format,
        "page_size": page_size,
        "total_size": num_tiles * page_size,
        "num_tiles": num_tiles,
        "block_count": int(cb.block_count),
    }


def _read_cpp_source(path: str) -> str:
    """Read a kernel cpp file; the bytes are embedded in the artifact so
    the flatbuffer survives ``/tmp`` cleanup and cross-machine moves."""
    with open(path, "r") as f:
        return f.read()


def _serialize_compiled_kernel(
    compiled,
    *,
    operand_metadata: Optional[dict] = None,
) -> bytes:
    """Turn a tt-lang ``CompiledTTNNKernel`` into a JSON byte blob.

    The schema is internal to tt-xla: produced here, consumed by the TTNN
    flatbuffer emitter in tt-mlir (``createProgramDescriptorFromTtLangArtifact``
    in ``lib/Target/TTNN/TTNNToFlatbuffer.cpp``). Keep both ends in sync
    via ``_ARTIFACT_FORMAT_VERSION`` (bump on any breaking change).

    The payload structure is::

        {
          "format_version": 1,
          "kernels": [
            {
              "thread_type":   "compute" | "noc",
              "cpp_source":    "<bytes of the kernel cpp file>",
              "tensor_indices":[<int>, ...],     # global tensor positions
              "kernel_config": { ... see _serialize_kernel_config ... }
            }, ...
          ],
          "core_range":  {"start": [x, y], "end": [x, y]},
          "cb_configs":  [{...}, ...],            # see _serialize_cb_config
          "num_tensors":      <int>,
          "num_pipe_nets":    <int>,
          "operand_metadata": { ... }             # see resolve_kernel
        }

    Every field is structured (no Python ``repr`` strings). The cpp
    sources are *embedded* directly, not referenced by path: this is
    what lets the flatbuffer be reloaded after ``/tmp`` is wiped or
    shipped to another machine.

    The artifact deliberately omits any ``tensor_accessor_args`` field.
    The TensorAccessor compile-time args are derived at execute time by
    the tt-mlir runtime from each operand's live ``Buffer*``; there is
    no correct value to bake at MLIR-translate time.
    """
    kernels: List[dict] = []
    noc_idx = 0
    for i, (kernel_path, thread_type) in enumerate(compiled.kernel_paths):
        cfg = compiled.kernel_configs[i]
        kernel_cfg = _serialize_kernel_config(thread_type, cfg, noc_idx)
        if thread_type == "noc":
            noc_idx += 1
        kernels.append(
            {
                "thread_type": thread_type,
                "cpp_source": _read_cpp_source(kernel_path),
                "tensor_indices": [int(t) for t in compiled.kernel_tensor_indices[i]],
                "kernel_config": kernel_cfg,
            }
        )

    payload = {
        "format_version": _ARTIFACT_FORMAT_VERSION,
        "kernels": kernels,
        "core_range": _serialize_core_range(compiled.core_ranges),
        "cb_configs": [_serialize_cb_config(c) for c in compiled.cb_configs],
        "num_tensors": int(compiled.num_tensors),
        "num_pipe_nets": int(compiled.num_pipe_nets),
    }
    if operand_metadata is not None:
        payload["operand_metadata"] = operand_metadata
    return json.dumps(payload).encode("utf-8")


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
    arg_roles: Optional[str] = None,
    shard_spec: Optional[str] = None,
    **_unused,
) -> bytes:
    """Resolve a registered kernel to a compiled artifact.

    Called by ``pjrt_plugin_tt::tt_lang_bridge::resolveKernels`` (under
    the GIL, via pybind11) for every ``ttnn.tt_lang_op`` in the
    post-TTNN module.

    Parameters
    ----------
    kernel_id, version_tag:
        Identifiers attached to the op by the StableHLO frontend. Must
        match the entry in the process-global registry; a mismatch means
        the kernel source was edited after the StableHLO program was
        cached.
    shapes, dtypes, layouts:
        Per-operand metadata, one entry per ``ttnn.tt_lang_op`` operand
        in declaration order. Shapes are the post-Shardy local-shard
        shapes. Dtype strings are MLIR-style ("f32", "bf16", ...).
        Layout strings are the printed ``ttnn.ttnn_layout<...>``
        encodings; the bridge sends ``""`` for operands without one.
    mesh_shape:
        Forwarded verbatim. tt-lang reads device topology from its own
        context, so this is recorded in the artifact for diagnostics
        but not passed into the compile path.

    Accepts any additional keyword arguments via ``**_unused`` so the
    plugin's bridge can grow new fields without breaking older Python
    side installs.

    Returns
    -------
    bytes
        JSON-encoded artifact understood by the TTNN flatbuffer emitter
        (see ``docs/source/tt_lang_integration.md`` for the schema and
        ``_serialize_compiled_kernel`` for the producer).
    """
    entry = get_registered_kernel(kernel_id)
    if entry.version_tag != version_tag:
        raise TtLangError(
            f"version_tag mismatch for kernel_id={kernel_id!r}: "
            f"registered={entry.version_tag!r}, requested={version_tag!r}. "
            f"This usually means the kernel source was edited after the "
            f"executable was compiled."
        )

    # Compile path is device-less: see the DEMO HACK banner above for
    # the upstream-fix motivation. NOC kernels' TensorAccessor
    # compile-time args are derived at launch time by the tt-mlir
    # runtime (``KernelArgTensorAccessorArgs`` markers) against the
    # live ``Buffer*``, so this side just hands tt-lang enough metadata
    # to JIT a kernel binary.
    args = _make_deviceless_ttnn_args(shapes, dtypes, layouts or [""] * len(shapes))

    compiled = _drive_ttl_compile(entry, args)
    return _serialize_compiled_kernel(
        compiled,
        operand_metadata={
            "shapes": [list(s) for s in shapes],
            "dtypes": list(dtypes),
            "layouts": list(layouts) if layouts else [""] * len(shapes),
            "mesh_shape": list(mesh_shape),
            "arg_roles": arg_roles,
            "shard_spec": shard_spec,
        },
    )
