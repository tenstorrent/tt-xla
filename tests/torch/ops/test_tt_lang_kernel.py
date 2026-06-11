# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
Unit tests for the tt-lang Python surface.

These tests cover what actually exists today: the ``torch.ops.tt.tt_lang_op``
custom op, the ``@tt_torch.tt_lang_operation`` decorator, the in-process operation
registry, and the Alt B ``resolve_operation`` entry point that drives tt-lang's
internal compile path. They do not require XLA hardware. A subset of the
``resolve_operation`` tests injects a fake ``ttl.ttl_api`` module via
``sys.modules`` so the rest of the wiring is exercised even when the real
tt-lang isn't installed in the test venv. End-to-end testing against the
real tt-lang and on-hardware execution belongs in hardware-gated integration
tests, added when the plugin/tt-mlir pieces described in
``docs/source/tt_lang_integration.md`` land.
"""

import json
import os
import sys
import types

import pytest
import torch
import tt_torch  # noqa: F401  -- ensures torch.ops.tt is populated
from tt_torch import tt_lang as tt_lang_mod
from tt_torch.custom_ops import _validate_tt_lang_op_out_indices
from tt_torch.tt_lang import (
    OperationEntry,
    TtLangError,
    tt_lang_operation,
    get_registered_operation,
    iter_registered_operations,
    resolve_operation,
)


@pytest.fixture
def clean_registry():
    """Snapshot/restore the global operation registry around a test."""
    saved = list(iter_registered_operations())
    tt_lang_mod._clear_registry_for_tests()
    try:
        yield
    finally:
        tt_lang_mod._clear_registry_for_tests()
        for entry in saved:
            tt_lang_mod._register(entry)


# ---------------------------------------------------------------------------
# torch.ops.tt.tt_lang_op
# ---------------------------------------------------------------------------


def test_torch_op_registered():
    """``torch.ops.tt.tt_lang_op`` must exist after importing tt_torch."""
    assert hasattr(torch.ops.tt, "tt_lang_op")
    assert callable(torch.ops.tt.tt_lang_op)


def test_torch_op_rejects_cpu_tensors():
    """The op is XLA-only; calling it on CPU must fail at dispatch."""
    a = torch.zeros(2, 3)
    # PyTorch's dispatcher raises NotImplementedError (or a subclass like
    # RuntimeError, depending on version) for an unregistered backend.
    with pytest.raises((NotImplementedError, RuntimeError)):
        torch.ops.tt.tt_lang_op([a], "k", "out", "vt0", "", [0])


def test_validate_out_indices_rejects_empty():
    with pytest.raises(ValueError, match="at least one"):
        _validate_tt_lang_op_out_indices([], num_tensors=1)


def test_validate_out_indices_rejects_out_of_range():
    with pytest.raises(ValueError, match="out of range"):
        _validate_tt_lang_op_out_indices([7], num_tensors=1)


def test_validate_out_indices_rejects_duplicates():
    with pytest.raises(ValueError, match="more than once"):
        _validate_tt_lang_op_out_indices([0, 0], num_tensors=2)


# ---------------------------------------------------------------------------
# Decorator + registry
# ---------------------------------------------------------------------------


def test_decorator_registers_operation(clean_registry):
    @tt_lang_operation(operation_id="unit.add.v1", arg_roles=("in", "in", "out"))
    def add(lhs, rhs, out): ...

    entry = get_registered_operation("unit.add.v1")
    assert isinstance(entry, OperationEntry)
    assert entry.operation_id == "unit.add.v1"
    assert entry.arg_roles == ("in", "in", "out")
    assert entry.version_tag
    assert callable(entry.impl)
    assert add._tt_lang_operation_entry is entry


def test_decorator_rejects_empty_operation_id(clean_registry):
    with pytest.raises(ValueError):

        @tt_lang_operation(operation_id="", arg_roles=("out",))
        def _bad(out): ...


def test_decorator_rejects_whitespace_in_operation_id(clean_registry):
    with pytest.raises(ValueError):

        @tt_lang_operation(operation_id="has space", arg_roles=("out",))
        def _bad(out): ...


def test_decorator_rejects_no_out_role(clean_registry):
    @tt_lang_operation(operation_id="unit.no_out.v1", arg_roles=("in", "in"))
    def k(lhs, rhs): ...

    a = torch.zeros(1)
    b = torch.zeros(1)
    with pytest.raises(ValueError):
        k(a, b)


def test_decorator_rejects_out_before_in(clean_registry):
    # ttnn.tt_lang_op is destination-passing style: arg_roles must list all
    # inputs before any output, so an "in" after an "out" is rejected at
    # decoration time.
    with pytest.raises(ValueError, match="before any"):

        @tt_lang_operation(operation_id="unit.bad_order.v1", arg_roles=("in", "out", "in"))
        def k(lhs, out, rhs): ...


def test_decorator_requires_explicit_arg_roles(clean_registry):
    # Without arg_roles=..., the decorator must fail loudly. Python
    # itself rejects the missing required keyword (TypeError); even if
    # someone passes arg_roles=None explicitly, _normalize_arg_roles
    # raises (ValueError). We accept either flavor of error.
    with pytest.raises((TypeError, ValueError)):

        @tt_lang_operation(operation_id="unit.no_roles.v1")  # type: ignore[call-arg]
        def k(lhs, rhs, out):  # pragma: no cover -- decoration raises
            ...

    with pytest.raises(ValueError):

        @tt_lang_operation(operation_id="unit.none_roles.v1", arg_roles=None)  # type: ignore[arg-type]
        def k2(lhs, rhs, out):  # pragma: no cover -- decoration raises
            ...


def test_decorator_rejects_double_registration_with_different_source(
    clean_registry,
):
    @tt_lang_operation(operation_id="unit.dup.v1", arg_roles=("in", "out"), version_tag="A")
    def k1(x, out): ...

    with pytest.raises(ValueError):

        @tt_lang_operation(operation_id="unit.dup.v1", arg_roles=("in", "out"), version_tag="B")
        def k2(x, out): ...


def test_decorator_double_registration_same_version_ok(clean_registry):
    """Re-registering with the same version_tag is idempotent."""

    @tt_lang_operation(operation_id="unit.idem.v1", arg_roles=("in", "out"), version_tag="A")
    def k1(x, out): ...

    @tt_lang_operation(operation_id="unit.idem.v1", arg_roles=("in", "out"), version_tag="A")
    def k2(x, out): ...

    entry = get_registered_operation("unit.idem.v1")
    assert entry.version_tag == "A"


def test_decorator_rejects_non_xla_tensors(clean_registry):
    # tt-lang has no CPU fallback by design (see decorator docstring) --
    # calling with CPU tensors must raise loudly so a passing CPU run
    # can never be confused with a working hardware kernel.
    @tt_lang_operation(operation_id="unit.cpu_no_ref.v1", arg_roles=("in", "out"))
    def k(x, out): ...

    x = torch.zeros(2)
    out = torch.zeros(2)
    with pytest.raises(NotImplementedError, match="XLA"):
        k(x, out)


def test_decorator_rejects_mixed_device(clean_registry):
    @tt_lang_operation(operation_id="unit.mixed.v1", arg_roles=("in", "out"))
    def k(x, out): ...

    class _FakeMeta(torch.Tensor):
        @property
        def device(self):
            return torch.device("meta")

    x_cpu = torch.zeros(2)
    out_meta = torch.zeros(2).as_subclass(_FakeMeta)
    with pytest.raises(ValueError):
        k(x_cpu, out_meta)


def test_iter_registered_operations_snapshot(clean_registry):
    @tt_lang_operation(operation_id="unit.iter.a.v1", arg_roles=("in", "out"))
    def a(x, out): ...

    @tt_lang_operation(operation_id="unit.iter.b.v1", arg_roles=("in", "out"))
    def b(x, out): ...

    ids = {e.operation_id for e in iter_registered_operations()}
    assert {"unit.iter.a.v1", "unit.iter.b.v1"} <= ids


# ---------------------------------------------------------------------------
# resolve_operation: registry / version-tag guards
# ---------------------------------------------------------------------------


def test_resolve_operation_unknown_id_raises(clean_registry):
    with pytest.raises(KeyError):
        resolve_operation(
            operation_id="unit.missing.v1",
            version_tag="vt0",
            shapes=[[1]],
            dtypes=["bfloat16"],
        )


def test_resolve_operation_version_tag_mismatch_raises(clean_registry):
    @tt_lang_operation(operation_id="unit.vtag.v1", arg_roles=("in", "out"), version_tag="vt0")
    def k(x, out): ...

    with pytest.raises(TtLangError):
        resolve_operation(
            operation_id="unit.vtag.v1",
            version_tag="stale-vt",
            shapes=[[1], [1]],
            dtypes=["bfloat16", "bfloat16"],
        )


# ---------------------------------------------------------------------------
# resolve_operation: Alt B compile-driver, exercised with a fake ttl module
# ---------------------------------------------------------------------------


class _MockCoreCoord:
    def __init__(self, x, y):
        self.x, self.y = x, y


class _MockCoreRange:
    def __init__(self, sx, sy, ex, ey):
        self.start_coord = _MockCoreCoord(sx, sy)
        self.end_coord = _MockCoreCoord(ex, ey)


class _MockCoreRangeSet:
    """Bare-minimum stand-in for ``ttnn.CoreRangeSet``.

    ``_serialize_core_range`` reads only ``.bounding_box()`` and the
    coords on the returned ``CoreRange``; mimicking that surface keeps
    the test independent of a real ttnn install.
    """

    def __init__(self, sx, sy, ex, ey):
        self._range = _MockCoreRange(sx, sy, ex, ey)

    def bounding_box(self):
        return self._range


class _MockTtnnDtype:
    """Mimics ``ttnn.DataType`` enum members.

    The serializer accepts any object whose ``.name`` matches the ttnn
    spelling (e.g. ``"BFLOAT16"``); see ``_ttnn_dtype_name``.
    """

    def __init__(self, name):
        self.name = name


class _MockRefTensor:
    def __init__(self, dtype_name):
        self.dtype = _MockTtnnDtype(dtype_name)


class _MockDataflowBuffer:
    """Stand-in for tt-lang's user-declared ``DataflowBuffer``.

    Exposes the four fields ``_serialize_cb_config`` reads: ``tensor``
    (for dtype), ``shape``, ``block_count``, and ``_cb_index``.
    """

    def __init__(self, cb_index, dtype_name="BFLOAT16", shape=(1, 1), block_count=2):
        self.tensor = _MockRefTensor(dtype_name)
        self.shape = shape
        self.block_count = block_count
        self._cb_index = cb_index


class _MockComputeConfig:
    """Mirror of ``ttnn.ComputeConfigDescriptor`` -- only the four bool /
    string knobs the serializer reads."""

    def __init__(self):
        self.math_fidelity = "HiFi4"
        self.fp32_dest_acc_en = False
        self.dst_full_sync_en = False
        self.bfp8_pack_precise = False
        self.math_approx_mode = False


class _FakeCompiledKernel:
    """Mimics the shape of tt-lang's ``CompiledTTNNKernel`` for tests.

    Holds only the fields ``_serialize_compiled_operation`` reads. Mock
    objects match the duck-typed shape of the real tt-lang/ttnn types,
    so the serializer exercises its real code paths.
    """

    def __init__(self, kernel_paths, num_tensors=3):
        self.kernel_paths = kernel_paths
        self.kernel_configs = [
            _MockComputeConfig(),
            object(),  # reader noc cfg: type is irrelevant once thread_type='noc'
            object(),  # writer noc cfg
        ]
        self.kernel_arg_specs = [[0, 1], [], []]
        self.cb_configs = [
            _MockDataflowBuffer(
                cb_index=0, dtype_name="BFLOAT16", shape=(1, 1), block_count=2
            ),
            _MockDataflowBuffer(
                cb_index=1, dtype_name="BFLOAT16", shape=(1, 1), block_count=2
            ),
            _MockDataflowBuffer(
                cb_index=2, dtype_name="BFLOAT16", shape=(1, 1), block_count=2
            ),
        ]
        self.core_ranges = _MockCoreRangeSet(0, 0, 0, 0)
        self.kernel_tensor_indices = [(0, 1, 2), (0, 1), (2,)]
        self.num_tensors = num_tensors
        self.num_pipe_nets = 1


@pytest.fixture
def fake_ttl(tmp_path):
    """Install a synthetic ``ttl.ttl_api`` module with ``_compile_kernel``.

    The fake records the env vars it observed at call time, so tests can
    verify ``TTLANG_COMPILE_ONLY=1`` was set while ``_compile_kernel``
    ran. Yields a ``state`` dict the test can introspect.

    The bridge has a single device-less compile path (DEMO HACK -- see
    ``tt_torch.tt_lang``); no chip is ever opened, no ttnn device is
    referenced, and TensorAccessor compile-time args are derived at
    runtime by the tt-mlir generic-op executor against the live
    ``Buffer*``.
    """
    compute_cpp = tmp_path / "compute.cpp"
    reader_cpp = tmp_path / "reader.cpp"
    writer_cpp = tmp_path / "writer.cpp"
    compute_cpp.write_text("// fake compute kernel\n")
    reader_cpp.write_text("// fake reader kernel\n")
    writer_cpp.write_text("// fake writer kernel\n")

    state = {
        "call_count": 0,
        "compile_only_during_call": None,
        "args_seen": None,
        "compile_result": _FakeCompiledKernel(
            kernel_paths=[
                (str(compute_cpp), "compute"),
                (str(reader_cpp), "noc"),
                (str(writer_cpp), "noc"),
            ]
        ),
        "raise": None,
    }

    def _fake_compile_kernel(*args, **kwargs):
        state["call_count"] += 1
        state["compile_only_during_call"] = os.environ.get("TTLANG_COMPILE_ONLY")
        state["args_seen"] = args
        if state["raise"] is not None:
            raise state["raise"]
        return state["compile_result"]

    fake_ttl_api = types.ModuleType("ttl.ttl_api")
    fake_ttl_api._compile_kernel = _fake_compile_kernel  # type: ignore[attr-defined]
    fake_ttl_pkg = types.ModuleType("ttl")
    fake_ttl_pkg.ttl_api = fake_ttl_api  # type: ignore[attr-defined]

    saved = {k: sys.modules.get(k) for k in ("ttl", "ttl.ttl_api")}
    sys.modules["ttl"] = fake_ttl_pkg
    sys.modules["ttl.ttl_api"] = fake_ttl_api
    try:
        yield state
    finally:
        for k, v in saved.items():
            if v is None:
                sys.modules.pop(k, None)
            else:
                sys.modules[k] = v


def _operation_that_calls_compile(operation_id):
    """Build a registered 3-arg operation whose ``impl`` forwards to the
    (mocked) tt-lang compile path. Mirrors how a real
    ``@ttl.operation @tt_lang_operation`` chain would look once compile is
    captured.
    """

    @tt_lang_operation(
        operation_id=operation_id,
        arg_roles=("in", "in", "out"),
        version_tag="vt0",
    )
    def impl(a, b, out):
        import ttl.ttl_api as _api

        _api._compile_kernel((a, b, out))

    return impl


def test_resolve_operation_returns_json_artifact(clean_registry, fake_ttl):
    """Artifact schema sanity check: structured kernels list with
    embedded cpp source bytes, plus structured core_range / cb_configs.
    """
    _operation_that_calls_compile("unit.resolve.basic.v1")

    artifact = resolve_operation(
        operation_id="unit.resolve.basic.v1",
        version_tag="vt0",
        shapes=[[1, 32, 32], [1, 32, 32], [1, 32, 32]],
        dtypes=["bf16", "bf16", "bf16"],
        layouts=("", "", ""),
        mesh_shape=(1,),
    )

    assert isinstance(artifact, bytes)
    payload = json.loads(artifact.decode("utf-8"))

    assert payload["format_version"] == tt_lang_mod._ARTIFACT_FORMAT_VERSION
    assert payload["num_tensors"] == 3
    assert payload["num_pipe_nets"] == 1
    assert payload["core_range"] == {"start": [0, 0], "end": [0, 0]}

    kernels = payload["kernels"]
    assert [k["thread_type"] for k in kernels] == ["compute", "noc", "noc"]
    # cpp source bytes are embedded, not paths.
    assert [k["cpp_source"] for k in kernels] == [
        "// fake compute kernel\n",
        "// fake reader kernel\n",
        "// fake writer kernel\n",
    ]
    # tensor_indices round-trip per-kernel.
    assert [k["tensor_indices"] for k in kernels] == [[0, 1, 2], [0, 1], [2]]
    # Reader/writer split is derived from noc kernel order.
    assert [k["kernel_config"]["type"] for k in kernels] == [
        "ComputeKernelConfig",
        "ReaderKernelConfig",
        "WriterKernelConfig",
    ]
    compute_cfg = kernels[0]["kernel_config"]
    assert compute_cfg["math_fidelity"] == "HiFi4"
    assert compute_cfg["fp32_dest_acc_en"] is False
    assert compute_cfg["dst_full_sync_en"] is False

    # CB configs are structured with byte sizes we can verify.
    cbs = payload["cb_configs"]
    assert len(cbs) == 3
    for i, cb in enumerate(cbs):
        assert cb["buffer_index"] == i
        assert cb["data_format"] == "BFloat16"
        assert cb["page_size"] == 32 * 32 * 2  # 2048 bytes/tile
        assert cb["num_tiles"] == 1 * 1 * 2  # shape (1,1) * block_count 2
        assert cb["total_size"] == cb["num_tiles"] * cb["page_size"]
        assert cb["block_count"] == 2

    assert payload["operand_metadata"]["shapes"] == [
        [1, 32, 32],
        [1, 32, 32],
        [1, 32, 32],
    ]
    assert payload["operand_metadata"]["dtypes"] == ["bf16", "bf16", "bf16"]
    assert payload["operand_metadata"]["mesh_shape"] == [1]


def test_resolve_operation_records_layout_metadata(clean_registry, fake_ttl):
    """Non-empty layout strings (post-TTNN encoding) must round-trip into
    the artifact so the flatbuffer emitter sees what was assumed.
    """
    _operation_that_calls_compile("unit.resolve.layouts.v1")

    layouts = (
        "#ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <1x1>, "
        "memref<1x1x!ttcore.tile<32x32, bf16>, #dram>, <interleaved>>",
        "#ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <1x1>, "
        "memref<1x1x!ttcore.tile<32x32, bf16>, #l1>, <interleaved>>",
        "",
    )
    artifact = resolve_operation(
        operation_id="unit.resolve.layouts.v1",
        version_tag="vt0",
        shapes=[[1, 32, 32], [1, 32, 32], [1, 32, 32]],
        dtypes=["bf16", "bf16", "bf16"],
        layouts=layouts,
        arg_roles="in,in,out",
        shard_spec='{"axis":0}',
    )
    payload = json.loads(artifact.decode("utf-8"))
    assert payload["operand_metadata"]["layouts"] == list(layouts)
    assert payload["operand_metadata"]["arg_roles"] == "in,in,out"
    assert payload["operand_metadata"]["shard_spec"] == '{"axis":0}'


def test_resolve_operation_defaults_empty_layouts(clean_registry, fake_ttl):
    """When no layouts are supplied the artifact records one empty string
    per operand (keeps the per-operand list length stable for consumers).
    """
    _operation_that_calls_compile("unit.resolve.default_layouts.v1")

    artifact = resolve_operation(
        operation_id="unit.resolve.default_layouts.v1",
        version_tag="vt0",
        shapes=[[1], [1], [1]],
        dtypes=["bf16", "bf16", "bf16"],
    )
    payload = json.loads(artifact.decode("utf-8"))
    assert payload["operand_metadata"]["layouts"] == ["", "", ""]
    assert payload["operand_metadata"]["arg_roles"] is None
    assert payload["operand_metadata"]["shard_spec"] is None


def test_resolve_operation_sets_compile_only_during_call(clean_registry, fake_ttl):
    """``TTLANG_COMPILE_ONLY=1`` must be active while ``_compile_kernel``
    runs, and the env var must be restored to its prior value after.
    """
    _operation_that_calls_compile("unit.resolve.compile_only.v1")

    saved = os.environ.pop("TTLANG_COMPILE_ONLY", None)
    try:
        resolve_operation(
            operation_id="unit.resolve.compile_only.v1",
            version_tag="vt0",
            shapes=[[1], [1], [1]],
            dtypes=["bf16", "bf16", "bf16"],
        )
        assert fake_ttl["compile_only_during_call"] == "1"
        assert "TTLANG_COMPILE_ONLY" not in os.environ
    finally:
        if saved is not None:
            os.environ["TTLANG_COMPILE_ONLY"] = saved


def test_resolve_operation_restores_compile_only_when_prev_set(clean_registry, fake_ttl):
    _operation_that_calls_compile("unit.resolve.prev_env.v1")

    os.environ["TTLANG_COMPILE_ONLY"] = "0"
    try:
        resolve_operation(
            operation_id="unit.resolve.prev_env.v1",
            version_tag="vt0",
            shapes=[[1], [1], [1]],
            dtypes=["bf16", "bf16", "bf16"],
        )
        assert fake_ttl["compile_only_during_call"] == "1"
        assert os.environ["TTLANG_COMPILE_ONLY"] == "0"
    finally:
        os.environ.pop("TTLANG_COMPILE_ONLY", None)


def test_resolve_operation_restores_monkey_patch_after_call(clean_registry, fake_ttl):
    _operation_that_calls_compile("unit.resolve.restore_patch.v1")

    import ttl.ttl_api as _api

    original = _api._compile_kernel
    resolve_operation(
        operation_id="unit.resolve.restore_patch.v1",
        version_tag="vt0",
        shapes=[[1], [1], [1]],
        dtypes=["bf16", "bf16", "bf16"],
    )
    assert _api._compile_kernel is original


def test_resolve_operation_restores_patch_on_compile_error(clean_registry, fake_ttl):
    _operation_that_calls_compile("unit.resolve.error_patch.v1")

    import ttl.ttl_api as _api

    original = _api._compile_kernel
    fake_ttl["raise"] = RuntimeError("simulated tt-lang failure")

    with pytest.raises(RuntimeError, match="simulated tt-lang failure"):
        resolve_operation(
            operation_id="unit.resolve.error_patch.v1",
            version_tag="vt0",
            shapes=[[1], [1], [1]],
            dtypes=["bf16", "bf16", "bf16"],
        )
    assert _api._compile_kernel is original
    assert "TTLANG_COMPILE_ONLY" not in os.environ


def test_resolve_operation_shape_dtype_count_mismatch(clean_registry, fake_ttl):
    _operation_that_calls_compile("unit.resolve.shape_mismatch.v1")

    with pytest.raises(TtLangError):
        resolve_operation(
            operation_id="unit.resolve.shape_mismatch.v1",
            version_tag="vt0",
            shapes=[[1], [1]],
            dtypes=["bf16", "bf16", "bf16"],
        )


def test_resolve_operation_raises_when_ttl_missing(clean_registry):
    """When the user's Python doesn't have tt-lang installed, surface a
    clear ``TtLangError`` rather than an opaque ``ImportError``.
    """

    @tt_lang_operation(operation_id="unit.no_ttl.v1", arg_roles=("in", "out"), version_tag="vt0")
    def k(x, out): ...

    saved = {k: sys.modules.get(k) for k in ("ttl", "ttl.ttl_api")}
    sys.modules["ttl"] = None  # type: ignore[assignment]
    sys.modules["ttl.ttl_api"] = None  # type: ignore[assignment]
    try:
        with pytest.raises(TtLangError, match="tt-lang is not importable"):
            resolve_operation(
                operation_id="unit.no_ttl.v1",
                version_tag="vt0",
                shapes=[[1], [1]],
                dtypes=["bf16", "bf16"],
            )
    finally:
        for k, v in saved.items():
            if v is None:
                sys.modules.pop(k, None)
            else:
                sys.modules[k] = v


# ---------------------------------------------------------------------------
# resolve_operation: device-less stub path (production default)
# ---------------------------------------------------------------------------


def test_resolve_operation_deviceless_passes_stub_tensors(clean_registry, fake_ttl):
    """Default (and only) path: ``resolve_operation`` feeds tt-lang
    ``_StubTtnnTensor`` stand-ins (DEMO HACK -- see ``tt_torch.tt_lang``
    for context).

    Asserts that the bridge:

    * never imports ttnn (the test fixture intentionally doesn't
      install a fake ttnn module);
    * passes stub tensors with the right shape/dtype/memory_config to
      ``ttl._compile_kernel`` (recorded by ``fake_ttl``);
    * does NOT serialise any per-operand ``tensor_accessor_args``
      field -- those are materialised at runtime by the tt-mlir
      generic-op executor from the live ``Buffer*``, not synthesised
      from metadata here.
    """
    from tt_torch.tt_lang import _StubTtnnTensor

    _operation_that_calls_compile("unit.resolve.deviceless.v1")

    artifact = resolve_operation(
        operation_id="unit.resolve.deviceless.v1",
        version_tag="vt0",
        shapes=[[2, 32, 32], [2, 32, 32], [2, 32, 32]],
        dtypes=["bf16", "f32", "i32"],
    )

    (stub_args,) = fake_ttl["args_seen"]
    assert len(stub_args) == 3
    assert all(isinstance(t, _StubTtnnTensor) for t in stub_args)
    # Stub layout must contain "TILE" so tt-lang's tilized check passes.
    assert all("TILE" in str(t.layout) for t in stub_args)
    # Memory config must classify as L1 or DRAM (default DRAM with no
    # layout string).
    for t in stub_args:
        assert str(t.memory_config().buffer_type) in ("L1", "DRAM")
    # Per-operand torch dtype mapping from the MLIR string still works,
    # since the stub carries the resolved torch.dtype.
    assert stub_args[0].dtype == torch.bfloat16
    assert stub_args[1].dtype == torch.float32
    assert stub_args[2].dtype == torch.int32
    for t in stub_args:
        assert tuple(t.shape) == (2, 32, 32)

    payload = json.loads(artifact.decode("utf-8"))
    # Contract: the JSON artifact does not carry any
    # ``tensor_accessor_args`` array. The tt-mlir flatbuffer emitter
    # writes ``KernelArgTensorAccessorArgs`` markers and the runtime
    # expands them against live buffers at launch.
    assert "tensor_accessor_args" not in payload
    assert payload["format_version"] == tt_lang_mod._ARTIFACT_FORMAT_VERSION


# Real-tt-lang compile integration is exercised in
# ``tests/torch/ops/test_tt_lang_kernel_e2e.py``, which drives the full
# stub-tensor compile path against the real tt-lang wheel and runs the
# resulting flatbuffer through tt-mlir on silicon. The e2e test is the
# single source of truth for the runtime TensorAccessor expansion path.
