# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Smoke + FX-level unit tests for the SwiGLU composite-fusion path.

Covers (cheap, no-silicon) the four pieces the composite-preservation
flow depends on:

* The 4 marker functions are registered in ``composite_ops.replacements``.
* The marker-decorated functions are not inlined by ``torch.fx.symbolic_trace``
  (i.e. ``@torch.fx.wrap`` is intact).
* ``SwiGLUFusionProvider`` is registered, default-disabled, and listed in
  ``DEFAULT_QUETZAL_REWRITE_PASSES`` so an `all` opt-in turns it on.
* ``run_selected_fusion_passes(["fuse_swiglu"])`` collapses both the
  ``narrow``-form and ``operator.getitem``-form decompositions of SwiGLU
  into a single ``_torch_swiglu`` marker call, AND accepts both
  ``silu(gate) * up`` and ``up * silu(gate)`` operand orders.
"""

import operator

import torch

import tt_torch.composite_ops as co
import tt_torch.fusion_providers as fp
from tt_torch.backend.passes import handle_composite_ops, run_selected_fusion_passes
from tt_torch.backend.quetzal_rewrite import (
    DEFAULT_QUETZAL_REWRITE_PASSES,
    get_quetzal_rewrite_passes,
)


# ---------- Composite registration ------------------------------------------


def test_swiglu_markers_in_replacements():
    """All 4 markers must be keyed in composite_ops.replacements so
    handle_composite_ops can swap them to their composite-emitting forms."""
    assert co._torch_swiglu in co.replacements
    assert co._torch_glu in co.replacements
    assert co._torch_geglu in co.replacements
    assert co._torch_reglu in co.replacements


def test_swiglu_markers_route_to_correct_composite_emitters():
    assert co.replacements[co._torch_swiglu] is co._composite_swiglu
    assert co.replacements[co._torch_glu] is co._composite_glu
    assert co.replacements[co._torch_geglu] is co._composite_geglu
    assert co.replacements[co._torch_reglu] is co._composite_reglu


def test_swiglu_marker_not_inlined_by_symbolic_trace():
    """``@torch.fx.wrap`` should keep the marker as an opaque call_function
    node — without that, FX would inline its body and we'd lose the
    fusion site."""

    def f(x):
        return co._torch_swiglu(x, dim=-1)

    gm = torch.fx.symbolic_trace(f)
    targets = [n.target for n in gm.graph.nodes if n.op == "call_function"]
    assert co._torch_swiglu in targets


# ---------- Provider registration -------------------------------------------


def test_swiglu_provider_registered_default_disabled():
    """``fuse_swiglu`` must appear in the default-disabled set, NOT the
    default-enabled set — silicon validation pending."""
    enabled = fp.FusionProvider.get_registered_provider_names()
    all_names = fp.FusionProvider.get_registered_provider_names(
        include_default_disabled=True
    )
    assert "fuse_swiglu" in all_names
    assert "fuse_swiglu" not in enabled


def test_swiglu_in_default_quetzal_rewrite_passes(monkeypatch):
    """``DEFAULT_QUETZAL_REWRITE_PASSES`` is the list activated by
    ``TT_TORCH_QUETZAL_REWRITE_PASSES=all``."""
    assert "fuse_swiglu" in DEFAULT_QUETZAL_REWRITE_PASSES

    monkeypatch.setenv("TT_TORCH_QUETZAL_REWRITE_PASSES", "all")
    passes = get_quetzal_rewrite_passes(None)
    assert "fuse_swiglu" in passes


# ---------- FX-level fusion ------------------------------------------------


def _swiglu_count(gm: torch.fx.GraphModule) -> int:
    return sum(
        1
        for n in gm.graph.nodes
        if n.op == "call_function" and n.target is co._torch_swiglu
    )


def _slice_count(gm: torch.fx.GraphModule) -> int:
    """Count any narrow / operator.getitem nodes (proxy for surviving
    SwiGLU half-slices)."""
    n = 0
    for node in gm.graph.nodes:
        if node.op == "call_method" and node.target == "narrow":
            n += 1
        if node.op == "call_function" and node.target is operator.getitem:
            n += 1
    return n


def test_fuse_swiglu_narrow_form():
    """The ``narrow``-spelled SwiGLU collapses to a single _torch_swiglu."""

    def swiglu_narrow(x: torch.Tensor) -> torch.Tensor:
        half = x.shape[-1] // 2
        gate = x.narrow(-1, 0, half)
        up = x.narrow(-1, half, half)
        return torch.nn.functional.silu(gate) * up

    gm = torch.fx.symbolic_trace(swiglu_narrow)
    replacements = run_selected_fusion_passes(gm, ["fuse_swiglu"])

    assert replacements == {"fuse_swiglu": 1}
    assert _swiglu_count(gm) == 1
    # Verify the result still computes correctly post-rewrite.
    x = torch.randn(2, 8)
    expected = torch.nn.functional.silu(x[..., :4]) * x[..., 4:]
    got = gm(x)
    torch.testing.assert_close(got, expected)


def test_fuse_swiglu_getitem_form():
    """``x[..., :half]`` form (operator.getitem with slice) also fuses."""

    def swiglu_getitem(x: torch.Tensor) -> torch.Tensor:
        half = x.shape[-1] // 2
        gate = x[..., :half]
        up = x[..., half:]
        return torch.nn.functional.silu(gate) * up

    gm = torch.fx.symbolic_trace(swiglu_getitem)
    replacements = run_selected_fusion_passes(gm, ["fuse_swiglu"])

    assert replacements == {"fuse_swiglu": 1}
    assert _swiglu_count(gm) == 1


def test_fuse_swiglu_reversed_mul_order():
    """``up * silu(gate)`` (operand order swapped) still matches."""

    def swiglu_reversed(x: torch.Tensor) -> torch.Tensor:
        half = x.shape[-1] // 2
        gate = x.narrow(-1, 0, half)
        up = x.narrow(-1, half, half)
        return up * torch.nn.functional.silu(gate)

    gm = torch.fx.symbolic_trace(swiglu_reversed)
    replacements = run_selected_fusion_passes(gm, ["fuse_swiglu"])

    assert replacements == {"fuse_swiglu": 1}
    assert _swiglu_count(gm) == 1


def test_fuse_swiglu_does_not_match_unrelated_mul():
    """A bare ``silu(x) * y`` with INDEPENDENT operands must NOT match —
    SwiGLU requires both halves to come from the SAME source tensor."""

    def not_swiglu(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return torch.nn.functional.silu(x) * y

    gm = torch.fx.symbolic_trace(not_swiglu)
    replacements = run_selected_fusion_passes(gm, ["fuse_swiglu"])

    assert replacements == {}
    assert _swiglu_count(gm) == 0


def test_handle_composite_ops_swaps_swiglu_marker_to_composite():
    """After fusion + handle_composite_ops, the _torch_swiglu marker node's
    target should be flipped to _composite_swiglu (the StableHLO-emitting
    wrapper)."""

    def swiglu(x: torch.Tensor) -> torch.Tensor:
        half = x.shape[-1] // 2
        return torch.nn.functional.silu(
            x.narrow(-1, 0, half)
        ) * x.narrow(-1, half, half)

    gm = torch.fx.symbolic_trace(swiglu)
    run_selected_fusion_passes(gm, ["fuse_swiglu"])
    handle_composite_ops(gm)

    targets = [n.target for n in gm.graph.nodes if n.op == "call_function"]
    assert co._composite_swiglu in targets
    assert co._torch_swiglu not in targets
