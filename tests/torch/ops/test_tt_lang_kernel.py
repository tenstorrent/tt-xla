# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
Unit tests for the tt-lang Python surface.

These tests cover the ``torch.ops.tt.tt_lang_op`` custom op, the
``@tt_torch.kernel`` decorator, and the in-process kernel registry. They
do not require XLA hardware. End-to-end testing against the real tt-lang
and on-hardware execution belongs in hardware-gated integration tests,
added when the plugin/tt-mlir pieces described in
``docs/src/tt_lang_integration.md`` land.
"""

import pytest
import torch

import tt_torch  # noqa: F401  -- ensures torch.ops.tt is populated
from tt_torch import tt_lang as tt_lang_mod
from tt_torch.custom_ops import _validate_tt_lang_op_out_indices
from tt_torch.tt_lang import (
    KernelEntry,
    get_registered_kernel,
    iter_registered_kernels,
    kernel,
)


@pytest.fixture
def clean_registry():
    """Snapshot/restore the global kernel registry around a test."""
    saved = list(iter_registered_kernels())
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


def test_decorator_registers_kernel(clean_registry):
    @kernel(kernel_id="unit.add.v1", arg_roles=("in", "in", "out"))
    def add(lhs, rhs, out):
        ...

    entry = get_registered_kernel("unit.add.v1")
    assert isinstance(entry, KernelEntry)
    assert entry.kernel_id == "unit.add.v1"
    assert entry.arg_roles == ("in", "in", "out")
    assert entry.version_tag
    assert callable(entry.impl)
    assert add._tt_lang_kernel_entry is entry


def test_decorator_rejects_empty_kernel_id(clean_registry):
    with pytest.raises(ValueError):

        @kernel(kernel_id="", arg_roles=("out",))
        def _bad(out):
            ...


def test_decorator_rejects_whitespace_in_kernel_id(clean_registry):
    with pytest.raises(ValueError):

        @kernel(kernel_id="has space", arg_roles=("out",))
        def _bad(out):
            ...


def test_decorator_rejects_no_out_role(clean_registry):
    @kernel(kernel_id="unit.no_out.v1", arg_roles=("in", "in"))
    def k(lhs, rhs):
        ...

    a = torch.zeros(1)
    b = torch.zeros(1)
    with pytest.raises(ValueError):
        k(a, b)


def test_decorator_requires_explicit_arg_roles(clean_registry):
    # Without arg_roles=..., the decorator must fail loudly. Python
    # itself rejects the missing required keyword (TypeError); even if
    # someone passes arg_roles=None explicitly, _normalize_arg_roles
    # raises (ValueError). We accept either flavor of error.
    with pytest.raises((TypeError, ValueError)):

        @kernel(kernel_id="unit.no_roles.v1")  # type: ignore[call-arg]
        def k(lhs, rhs, out):  # pragma: no cover -- decoration raises
            ...

    with pytest.raises(ValueError):

        @kernel(kernel_id="unit.none_roles.v1", arg_roles=None)  # type: ignore[arg-type]
        def k2(lhs, rhs, out):  # pragma: no cover -- decoration raises
            ...


def test_decorator_rejects_double_registration_with_different_source(
    clean_registry,
):
    @kernel(kernel_id="unit.dup.v1", arg_roles=("in", "out"), version_tag="A")
    def k1(x, out):
        ...

    with pytest.raises(ValueError):

        @kernel(kernel_id="unit.dup.v1", arg_roles=("in", "out"), version_tag="B")
        def k2(x, out):
            ...


def test_decorator_double_registration_same_version_ok(clean_registry):
    """Re-registering with the same version_tag is idempotent."""

    @kernel(kernel_id="unit.idem.v1", arg_roles=("in", "out"), version_tag="A")
    def k1(x, out):
        ...

    @kernel(kernel_id="unit.idem.v1", arg_roles=("in", "out"), version_tag="A")
    def k2(x, out):
        ...

    entry = get_registered_kernel("unit.idem.v1")
    assert entry.version_tag == "A"


def test_decorator_rejects_non_xla_tensors(clean_registry):
    # tt-lang has no CPU fallback by design (see decorator docstring) --
    # calling with CPU tensors must raise loudly so a passing CPU run
    # can never be confused with a working hardware kernel.
    @kernel(kernel_id="unit.cpu_no_ref.v1", arg_roles=("in", "out"))
    def k(x, out):
        ...

    x = torch.zeros(2)
    out = torch.zeros(2)
    with pytest.raises(NotImplementedError, match="XLA"):
        k(x, out)


def test_decorator_rejects_mixed_device(clean_registry):
    @kernel(kernel_id="unit.mixed.v1", arg_roles=("in", "out"))
    def k(x, out):
        ...

    class _FakeMeta(torch.Tensor):
        @property
        def device(self):
            return torch.device("meta")

    x_cpu = torch.zeros(2)
    out_meta = torch.zeros(2).as_subclass(_FakeMeta)
    with pytest.raises(ValueError):
        k(x_cpu, out_meta)


def test_iter_registered_kernels_snapshot(clean_registry):
    @kernel(kernel_id="unit.iter.a.v1", arg_roles=("in", "out"))
    def a(x, out):
        ...

    @kernel(kernel_id="unit.iter.b.v1", arg_roles=("in", "out"))
    def b(x, out):
        ...

    ids = {e.kernel_id for e in iter_registered_kernels()}
    assert {"unit.iter.a.v1", "unit.iter.b.v1"} <= ids
