# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Smoke + FX-level unit tests for the slice_reshape composite-fusion path.

Covers (cheap, no-silicon) the four pieces the composite-preservation flow
depends on:

* The marker function ``_torch_slice_reshape`` is registered in
  ``composite_ops.replacements`` and routes to ``_composite_slice_reshape``.
* The marker is not inlined by ``torch.fx.symbolic_trace`` (the
  ``@torch.fx.wrap`` decorator is intact).
* ``SliceReshapeFusionProvider`` is registered, default-disabled, and
  listed in ``DEFAULT_QUETZAL_REWRITE_PASSES`` so an "all" opt-in turns it
  on.
* ``run_selected_fusion_passes(["fuse_slice_reshape"])`` collapses
  ``narrow``-form slice + reshape chains into a single
  ``_torch_slice_reshape`` marker call AND rejects multi-user slices and
  rank-dropping ``getitem`` slices (mirroring the quetzal pass exactly).
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


def test_slice_reshape_marker_in_replacements():
    """Marker must be keyed in composite_ops.replacements so
    handle_composite_ops can swap it to the composite-emitting form."""
    assert co._torch_slice_reshape in co.replacements


def test_slice_reshape_marker_routes_to_composite_emitter():
    assert co.replacements[co._torch_slice_reshape] is co._composite_slice_reshape


def test_slice_reshape_marker_not_inlined_by_symbolic_trace():
    """``@torch.fx.wrap`` should keep the marker as an opaque
    call_function node — without that, FX would inline its body and we'd
    lose the fusion site."""

    def f(x):
        return co._torch_slice_reshape(x, [(0, 0, 1)], [4])

    gm = torch.fx.symbolic_trace(f)
    targets = [n.target for n in gm.graph.nodes if n.op == "call_function"]
    assert co._torch_slice_reshape in targets


# ---------- Provider registration -------------------------------------------


def test_slice_reshape_provider_registered_default_disabled():
    """``fuse_slice_reshape`` must appear in the default-disabled set, NOT
    the default-enabled set — silicon validation pending."""
    enabled = fp.FusionProvider.get_registered_provider_names()
    all_names = fp.FusionProvider.get_registered_provider_names(
        include_default_disabled=True
    )
    assert "fuse_slice_reshape" in all_names
    assert "fuse_slice_reshape" not in enabled


def test_slice_reshape_in_default_quetzal_rewrite_passes(monkeypatch):
    """``DEFAULT_QUETZAL_REWRITE_PASSES`` is the list activated by
    ``TT_TORCH_QUETZAL_REWRITE_PASSES=all``."""
    assert "fuse_slice_reshape" in DEFAULT_QUETZAL_REWRITE_PASSES

    monkeypatch.setenv("TT_TORCH_QUETZAL_REWRITE_PASSES", "all")
    passes = get_quetzal_rewrite_passes(None)
    assert "fuse_slice_reshape" in passes


# ---------- FX-level fusion ------------------------------------------------


def _slice_reshape_count(gm: torch.fx.GraphModule) -> int:
    return sum(
        1
        for n in gm.graph.nodes
        if n.op == "call_function" and n.target is co._torch_slice_reshape
    )


def _narrow_count(gm: torch.fx.GraphModule) -> int:
    return sum(
        1
        for n in gm.graph.nodes
        if n.op == "call_method" and n.target == "narrow"
    )


def _reshape_count(gm: torch.fx.GraphModule) -> int:
    """Count any reshape / view nodes left after the rewrite."""
    n = 0
    for node in gm.graph.nodes:
        if node.op == "call_method" and node.target in ("reshape", "view"):
            n += 1
        if (
            node.op == "call_function"
            and getattr(node.target, "__name__", None) in ("reshape", "view")
        ):
            n += 1
    return n


def test_fuse_slice_reshape_narrow_form():
    """The ``narrow``-spelled slice followed by a reshape collapses to a
    single _torch_slice_reshape call. Mirrors the canonical Quetzal QKV
    pattern: slice(:2048) along last dim then reshape into multi-head.
    """

    def slice_reshape_narrow(x: torch.Tensor) -> torch.Tensor:
        sliced = x.narrow(-1, 0, 2048)
        return sliced.reshape(1, 32, 64)

    gm = torch.fx.symbolic_trace(slice_reshape_narrow)
    replacements = run_selected_fusion_passes(gm, ["fuse_slice_reshape"])

    assert replacements == {"fuse_slice_reshape": 1}
    assert _slice_reshape_count(gm) == 1
    # Narrow + reshape should both be gone (or at least no longer the
    # primary path; the marker call replaces them).
    assert _narrow_count(gm) == 0
    assert _reshape_count(gm) == 0
    # Verify the result still computes correctly post-rewrite.
    x = torch.randn(1, 1, 6144)
    expected = x.narrow(-1, 0, 2048).reshape(1, 32, 64)
    got = gm(x)
    torch.testing.assert_close(got, expected)


def test_fuse_slice_reshape_chained_narrow():
    """Multiple narrows (different dims) in a chain followed by one
    reshape — encode as one composite with full per-dim begins/ends."""

    def slice_reshape_chain(x: torch.Tensor) -> torch.Tensor:
        sliced = x.narrow(0, 0, 1).narrow(2, 0, 2048)
        return sliced.reshape(32, 64)

    gm = torch.fx.symbolic_trace(slice_reshape_chain)
    replacements = run_selected_fusion_passes(gm, ["fuse_slice_reshape"])

    assert replacements == {"fuse_slice_reshape": 1}
    assert _slice_reshape_count(gm) == 1


def test_fuse_slice_reshape_rejects_multi_user_slice():
    """If the slice has TWO consumers (the reshape we want and another
    op), the quetzal matcher rejects — slice must be exclusively consumed
    by the reshape, otherwise we'd orphan the other consumer's data."""

    def two_consumers(x: torch.Tensor) -> torch.Tensor:
        sliced = x.narrow(-1, 0, 2048)
        return sliced.reshape(1, 32, 64) + sliced.sum()

    gm = torch.fx.symbolic_trace(two_consumers)
    replacements = run_selected_fusion_passes(gm, ["fuse_slice_reshape"])

    assert replacements == {}
    assert _slice_reshape_count(gm) == 0


def test_fuse_slice_reshape_rejects_rank_dropping_getitem():
    """``x[0]`` (rank-dropping getitem) is NOT a same-rank slice — quetzal
    requires same-rank, so we don't fuse this case."""

    def drop_rank(x: torch.Tensor) -> torch.Tensor:
        return x[0].reshape(32, 64)

    gm = torch.fx.symbolic_trace(drop_rank)
    replacements = run_selected_fusion_passes(gm, ["fuse_slice_reshape"])

    assert replacements == {}
    assert _slice_reshape_count(gm) == 0


def test_handle_composite_ops_swaps_slice_reshape_marker_to_composite():
    """After fusion + handle_composite_ops, the _torch_slice_reshape
    marker node's target should be flipped to _composite_slice_reshape
    (the StableHLO-emitting wrapper)."""

    def f(x: torch.Tensor) -> torch.Tensor:
        return x.narrow(-1, 0, 2048).reshape(1, 32, 64)

    gm = torch.fx.symbolic_trace(f)
    run_selected_fusion_passes(gm, ["fuse_slice_reshape"])
    handle_composite_ops(gm)

    targets = [n.target for n in gm.graph.nodes if n.op == "call_function"]
    assert co._composite_slice_reshape in targets
    assert co._torch_slice_reshape not in targets
