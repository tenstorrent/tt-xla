# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Unit tests for the ``erase_repeat_kv`` FX pass (tt_torch/backend/passes.py).

The pass rewrites the key/value operands of ``scaled_dot_product_attention`` from
their grouped-query ``repeat_kv`` head-expansion to the non-inflated source and
sets ``enable_gqa=True`` instead, so the downstream (composite / ttnn) SDPA can
broadcast the KV heads on device.

These are pure graph-transformation tests: they dynamo-capture a module's forward
graph, run the pass on it directly, and assert on both the resulting graph
structure (what was / wasn't rewritten) and numerical equivalence against eager.
No Tenstorrent device is required. The SDPA math backend is pinned so eager and
transformed executions use the same kernel and compare bit-for-bit.
"""

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_xla  # noqa: F401  # pass import chain registers xla custom ops
from torch.nn.attention import SDPBackend, sdpa_kernel
from tt_torch.backend.passes import (
    _SDPA_HEAD_AXIS_FROM_END,
    _fx_node_shape,
    _normalize_dim,
    _sdpa_operand,
    count_inplace_mutations,
    erase_repeat_kv,
)
from utils import Category

SDPA = torch.nn.functional.scaled_dot_product_attention


# ---------------------------------------------------------------------------
# Helpers: repeat_kv variants
# ---------------------------------------------------------------------------
def repeat_kv_hf(hidden: torch.Tensor, n_rep: int) -> torch.Tensor:
    """HF-style repeat_kv: unsqueeze-after-head -> expand -> reshape (interleaved)."""
    b, h, s, d = hidden.shape
    if n_rep == 1:
        return hidden
    hidden = hidden[:, :, None, :, :].expand(b, h, n_rep, s, d)
    return hidden.reshape(b, h * n_rep, s, d)


def repeat_kv_explicit_unsqueeze(hidden: torch.Tensor, n_rep: int) -> torch.Tensor:
    """repeat_kv using an explicit .unsqueeze(head+1) instead of None-indexing.

    Exercises the second unsqueeze form in _match_unsqueeze_after; the None-index
    form in repeat_kv_hf exercises the first.
    """
    b, h, s, d = hidden.shape
    if n_rep == 1:
        return hidden
    hidden = hidden.unsqueeze(2).expand(b, h, n_rep, s, d)
    return hidden.reshape(b, h * n_rep, s, d)


def repeat_kv_tile_via_unsqueeze(hidden: torch.Tensor, n_rep: int) -> torch.Tensor:
    """Tile-style expansion: unsqueeze *before* head -> expand -> reshape.

    Produces block ordering (head = r*Hkv + hkv), which is NOT what enable_gqa
    does, so the pass must refuse to rewrite it.
    """
    b, h, s, d = hidden.shape
    hidden = hidden[:, None, :, :, :].expand(b, n_rep, h, s, d)
    return hidden.reshape(b, n_rep * h, s, d)


# ---------------------------------------------------------------------------
# Helpers: capture + run
# ---------------------------------------------------------------------------
def _count(gm, op, target) -> int:
    return sum(1 for n in gm.graph.nodes if n.op == op and n.target == target)


def _count_repeat_interleave(gm) -> int:
    return sum(
        1
        for n in gm.graph.nodes
        if (n.op == "call_function" and n.target is torch.repeat_interleave)
        or (n.op == "call_method" and n.target == "repeat_interleave")
    )


def _sdpa_nodes(gm):
    return [n for n in gm.graph.nodes if n.op == "call_function" and n.target is SDPA]


def _kv_head_counts(gm):
    """Return list of (key_heads, value_heads) per SDPA node."""
    out = []
    for sn in _sdpa_nodes(gm):
        k = _fx_node_shape(_sdpa_operand(sn, 1, "key"))
        v = _fx_node_shape(_sdpa_operand(sn, 2, "value"))
        out.append((k[-_SDPA_HEAD_AXIS_FROM_END], v[-_SDPA_HEAD_AXIS_FROM_END]))
    return out


def _enable_gqa_flags(gm):
    return [sn.kwargs.get("enable_gqa", None) for sn in _sdpa_nodes(gm)]


def _kv_operands_are_rewired(gm):
    """True if every SDPA key/value operand is a plain source, not a repeat_kv node.

    Confirms the operand was rewired past the expansion, independent of whether the
    now-dead expansion node has been eliminated (the pass intentionally leaves DCE
    to the mutation-aware export pipeline).
    """
    repeat_targets = {"reshape", "view", "expand", "repeat_interleave"}
    for sn in _sdpa_nodes(gm):
        for pos, kw in ((1, "key"), (2, "value")):
            op = _sdpa_operand(sn, pos, kw)
            if op.op == "call_method" and op.target in repeat_targets:
                return False
            if op.op == "call_function" and op.target is torch.repeat_interleave:
                return False
    return True


def run_pass(module, inputs):
    """Dynamo-capture module.forward, apply erase_repeat_kv, assert numerics, return info.

    The pass rewires SDPA operands but intentionally does NOT eliminate dead nodes
    (that is left to the mutation-aware export pipeline), so we assert on the SDPA
    operands rather than on global expansion-node counts. `*_before` counts confirm
    the repeat_kv pattern was actually present pre-pass.
    """
    torch._dynamo.reset()
    info = {}

    def backend(gm, example_inputs):
        info["expand_before"] = _count(gm, "call_method", "expand")
        info["ri_before"] = _count_repeat_interleave(gm)
        info["mutations_before"] = count_inplace_mutations(gm)
        erase_repeat_kv(gm)
        info["mutations_after"] = count_inplace_mutations(gm)
        info["gm"] = gm
        return gm.forward

    # Pin the SDPA math backend so eager and transformed use the same kernel and
    # the enable_gqa substitution compares exactly.
    with sdpa_kernel(SDPBackend.MATH):
        eager = module(*inputs)
        compiled = torch.compile(module, backend=backend, fullgraph=True)
        transformed = compiled(*inputs)

    torch.testing.assert_close(transformed, eager)
    return info


def _qkv(batch, q_heads, kv_heads, seq, dim, dtype=torch.float32):
    q = torch.randn(batch, q_heads, seq, dim, dtype=dtype)
    k = torch.randn(batch, kv_heads, seq, dim, dtype=dtype)
    v = torch.randn(batch, kv_heads, seq, dim, dtype=dtype)
    return q, k, v


# ===========================================================================
# Positive cases: repeat_kv should be erased
# ===========================================================================
@pytest.mark.push
@pytest.mark.nightly
@pytest.mark.single_device
@pytest.mark.record_test_properties(category=Category.GRAPH_TEST)
@pytest.mark.parametrize("n_rep", [2, 4, 8])
@pytest.mark.parametrize("kv_heads", [1, 2])
def test_hf_repeat_kv_is_erased(n_rep, kv_heads):
    q_heads = kv_heads * n_rep

    class M(nn.Module):
        def forward(self, q, k, v):
            return SDPA(q, repeat_kv_hf(k, n_rep), repeat_kv_hf(v, n_rep))

    q, k, v = _qkv(1, q_heads, kv_heads, 16, 32)
    info = run_pass(M(), (q, k, v))

    # The repeat_kv pattern was present pre-pass...
    assert info["expand_before"] == 2
    # ...and SDPA now consumes non-inflated K/V (rewired past the expansion) and
    # is told to broadcast heads on device.
    assert _kv_operands_are_rewired(info["gm"])
    assert _kv_head_counts(info["gm"]) == [(kv_heads, kv_heads)]
    assert _enable_gqa_flags(info["gm"]) == [True]


@pytest.mark.push
@pytest.mark.nightly
@pytest.mark.single_device
@pytest.mark.record_test_properties(category=Category.GRAPH_TEST)
def test_repeat_interleave_form_is_erased():
    kv_heads, n_rep = 2, 4

    class M(nn.Module):
        def forward(self, q, k, v):
            return SDPA(
                q,
                k.repeat_interleave(n_rep, dim=-3),
                v.repeat_interleave(n_rep, dim=-3),
            )

    q, k, v = _qkv(1, kv_heads * n_rep, kv_heads, 16, 32)
    info = run_pass(M(), (q, k, v))

    assert info["ri_before"] == 2
    assert _kv_operands_are_rewired(info["gm"])
    assert _kv_head_counts(info["gm"]) == [(kv_heads, kv_heads)]
    assert _enable_gqa_flags(info["gm"]) == [True]


@pytest.mark.push
@pytest.mark.nightly
@pytest.mark.single_device
@pytest.mark.record_test_properties(category=Category.GRAPH_TEST)
def test_explicit_unsqueeze_repeat_kv_is_erased():
    """repeat_kv built with .unsqueeze() rather than None-indexing (matcher form 2)."""
    kv_heads, n_rep = 2, 4

    class M(nn.Module):
        def forward(self, q, k, v):
            return SDPA(
                q,
                repeat_kv_explicit_unsqueeze(k, n_rep),
                repeat_kv_explicit_unsqueeze(v, n_rep),
            )

    q, k, v = _qkv(1, kv_heads * n_rep, kv_heads, 16, 32)
    info = run_pass(M(), (q, k, v))

    assert info["expand_before"] == 2
    assert _kv_operands_are_rewired(info["gm"])
    assert _kv_head_counts(info["gm"]) == [(kv_heads, kv_heads)]
    assert _enable_gqa_flags(info["gm"]) == [True]


@pytest.mark.push
@pytest.mark.nightly
@pytest.mark.single_device
@pytest.mark.record_test_properties(category=Category.GRAPH_TEST)
def test_attn_mask_and_scale_are_preserved():
    kv_heads, n_rep, seq = 2, 4, 16
    q_heads = kv_heads * n_rep

    class M(nn.Module):
        def forward(self, q, k, v, mask):
            return SDPA(
                q,
                repeat_kv_hf(k, n_rep),
                repeat_kv_hf(v, n_rep),
                attn_mask=mask,
                scale=0.123,
            )

    q, k, v = _qkv(1, q_heads, kv_heads, seq, 32)
    mask = torch.zeros(1, q_heads, seq, seq, dtype=torch.float32)
    info = run_pass(M(), (q, k, v, mask))

    gm = info["gm"]
    assert _kv_operands_are_rewired(gm)
    assert _kv_head_counts(gm) == [(kv_heads, kv_heads)]
    assert _enable_gqa_flags(gm) == [True]
    # attn_mask and scale must survive the rewrite untouched.
    (sn,) = _sdpa_nodes(gm)
    assert sn.kwargs.get("scale") == 0.123
    mask_operand = sn.kwargs.get("attn_mask")
    assert mask_operand is not None
    assert _fx_node_shape(mask_operand) == (1, q_heads, seq, seq)


@pytest.mark.push
@pytest.mark.nightly
@pytest.mark.single_device
@pytest.mark.record_test_properties(category=Category.GRAPH_TEST)
def test_keyword_key_value_operands_are_erased():
    kv_heads, n_rep = 2, 4

    class M(nn.Module):
        def forward(self, q, k, v):
            # key/value passed by keyword -> exercises the kwargs write-back path.
            return SDPA(q, key=repeat_kv_hf(k, n_rep), value=repeat_kv_hf(v, n_rep))

    q, k, v = _qkv(1, kv_heads * n_rep, kv_heads, 16, 32)
    info = run_pass(M(), (q, k, v))

    assert _kv_operands_are_rewired(info["gm"])
    assert _kv_head_counts(info["gm"]) == [(kv_heads, kv_heads)]
    assert _enable_gqa_flags(info["gm"]) == [True]


@pytest.mark.push
@pytest.mark.nightly
@pytest.mark.single_device
@pytest.mark.record_test_properties(category=Category.GRAPH_TEST)
def test_multiple_attention_layers_all_erased():
    kv_heads, n_rep = 2, 4
    q_heads = kv_heads * n_rep

    class M(nn.Module):
        def forward(self, q0, k0, v0, q1, k1, v1):
            a = SDPA(q0, repeat_kv_hf(k0, n_rep), repeat_kv_hf(v0, n_rep))
            b = SDPA(q1, repeat_kv_hf(k1, n_rep), repeat_kv_hf(v1, n_rep))
            return a + b

    q0, k0, v0 = _qkv(1, q_heads, kv_heads, 16, 32)
    q1, k1, v1 = _qkv(1, q_heads, kv_heads, 16, 32)
    info = run_pass(M(), (q0, k0, v0, q1, k1, v1))

    assert info["expand_before"] == 4
    assert _kv_operands_are_rewired(info["gm"])
    assert _kv_head_counts(info["gm"]) == [(kv_heads, kv_heads), (kv_heads, kv_heads)]
    assert _enable_gqa_flags(info["gm"]) == [True, True]


@pytest.mark.push
@pytest.mark.nightly
@pytest.mark.single_device
@pytest.mark.record_test_properties(category=Category.GRAPH_TEST)
def test_shared_repeat_kv_keeps_other_consumer():
    """If the inflated K/V also feeds a non-SDPA consumer, only SDPA is rewired."""
    kv_heads, n_rep = 2, 4
    q_heads = kv_heads * n_rep

    class M(nn.Module):
        def forward(self, q, k, v):
            k_inflated = repeat_kv_hf(k, n_rep)
            out = SDPA(q, k_inflated, repeat_kv_hf(v, n_rep))
            # k_inflated is also returned -> the shared expansion must stay intact
            # for the second consumer while SDPA is rewired past it.
            return out, k_inflated

    q, k, v = _qkv(1, q_heads, kv_heads, 16, 32)
    info = run_pass(M(), (q, k, v))

    gm = info["gm"]
    # SDPA operand is rewired to the non-inflated key and enable_gqa is set...
    assert _kv_operands_are_rewired(gm)
    assert _kv_head_counts(gm) == [(kv_heads, kv_heads)]
    assert _enable_gqa_flags(gm) == [True]
    # ...while the returned k_inflated (32 heads) is preserved -- verified by the
    # numerics check in run_pass, which compares the full (out, k_inflated) tuple.


@pytest.mark.push
@pytest.mark.nightly
@pytest.mark.single_device
@pytest.mark.record_test_properties(category=Category.GRAPH_TEST)
def test_kv_cache_copy_mutation_is_preserved():
    """Regression: the pass must not drop in-place KV-cache writes (Tensor.copy_).

    A prefill KV-cache fill is a `call_method copy_` that torch.fx treats as pure, so
    a blanket eliminate_dead_code() in the pass would delete it once SDPA is rewired
    past the repeat_kv (observed as PCC ~0.3, all fill_cache/RoPE ops removed). The
    pass must rewire SDPA yet leave the mutation intact for torch.export to
    functionalize.
    """
    kv_heads, n_rep = 2, 4
    q_heads = kv_heads * n_rep

    class M(nn.Module):
        def __init__(self):
            super().__init__()
            self.register_buffer("kcache", torch.zeros(1, kv_heads, 16, 32))

        def forward(self, q, k, v):
            self.kcache.copy_(k)  # in-place cache fill -> call_method copy_
            return SDPA(q, repeat_kv_hf(self.kcache, n_rep), repeat_kv_hf(v, n_rep))

    q, k, v = _qkv(1, q_heads, kv_heads, 16, 32)
    info = run_pass(M(), (q, k, v))

    # The cache write must survive the pass (this is the actual PCC=0.3 regression).
    assert info["mutations_before"] >= 1
    assert info["mutations_after"] == info["mutations_before"]
    # SDPA is still rewired to the non-inflated cache tensor + enable_gqa.
    assert _kv_operands_are_rewired(info["gm"])
    assert _kv_head_counts(info["gm"]) == [(kv_heads, kv_heads)]
    assert _enable_gqa_flags(info["gm"]) == [True]


@pytest.mark.push
@pytest.mark.nightly
@pytest.mark.single_device
@pytest.mark.record_test_properties(category=Category.GRAPH_TEST)
def test_count_inplace_mutations_counts_copy():
    """count_inplace_mutations detects trailing-underscore in-place call_method ops."""
    torch._dynamo.reset()
    captured = {}

    class M(nn.Module):
        def __init__(self):
            super().__init__()
            self.register_buffer("buf", torch.zeros(4))

        def forward(self, x):
            self.buf.copy_(x)
            return x + 1

    def backend(gm, example_inputs):
        captured["count"] = count_inplace_mutations(gm)
        return gm.forward

    torch.compile(M(), backend=backend, fullgraph=True)(torch.randn(4))
    assert captured["count"] >= 1


# ===========================================================================
# Negative cases: pass must NOT rewrite
# ===========================================================================
@pytest.mark.push
@pytest.mark.nightly
@pytest.mark.single_device
@pytest.mark.record_test_properties(category=Category.GRAPH_TEST)
def test_plain_mha_untouched():
    heads = 8

    class M(nn.Module):
        def forward(self, q, k, v):
            return SDPA(q, k, v)

    q, k, v = _qkv(1, heads, heads, 16, 32)
    info = run_pass(M(), (q, k, v))

    assert _kv_head_counts(info["gm"]) == [(heads, heads)]
    # No expansion existed, so enable_gqa is never introduced.
    assert _enable_gqa_flags(info["gm"]) == [None]


@pytest.mark.push
@pytest.mark.nightly
@pytest.mark.single_device
@pytest.mark.record_test_properties(category=Category.GRAPH_TEST)
def test_tile_repeat_not_erased():
    """torch.repeat / tile ordering differs from enable_gqa -> must not be rewritten."""
    kv_heads, n_rep = 2, 4
    q_heads = kv_heads * n_rep

    class M(nn.Module):
        def forward(self, q, k, v):
            return SDPA(q, k.repeat(1, n_rep, 1, 1), v.repeat(1, n_rep, 1, 1))

    q, k, v = _qkv(1, q_heads, kv_heads, 16, 32)
    info = run_pass(M(), (q, k, v))

    # K/V stay inflated and enable_gqa is not set.
    assert _kv_head_counts(info["gm"]) == [(q_heads, q_heads)]
    assert _enable_gqa_flags(info["gm"]) == [None]


@pytest.mark.push
@pytest.mark.nightly
@pytest.mark.single_device
@pytest.mark.record_test_properties(category=Category.GRAPH_TEST)
def test_unsqueeze_before_head_not_erased():
    """expand+reshape with the rep dim inserted before the head axis is tile order."""
    kv_heads, n_rep = 2, 4
    q_heads = kv_heads * n_rep

    class M(nn.Module):
        def forward(self, q, k, v):
            return SDPA(
                q,
                repeat_kv_tile_via_unsqueeze(k, n_rep),
                repeat_kv_tile_via_unsqueeze(v, n_rep),
            )

    q, k, v = _qkv(1, q_heads, kv_heads, 16, 32)
    info = run_pass(M(), (q, k, v))

    assert _kv_head_counts(info["gm"]) == [(q_heads, q_heads)]
    assert _enable_gqa_flags(info["gm"]) == [None]


@pytest.mark.push
@pytest.mark.nightly
@pytest.mark.single_device
@pytest.mark.record_test_properties(category=Category.GRAPH_TEST)
def test_repeat_interleave_on_wrong_axis_not_erased():
    """repeat_interleave along a non-head axis must not be treated as repeat_kv."""
    heads, n_rep = 8, 2

    class M(nn.Module):
        def forward(self, q, k, v):
            # Interleave along seq (dim=-2), not heads -> different semantics.
            return SDPA(q, k.repeat_interleave(n_rep, dim=-2), v)

    q = torch.randn(1, heads, 16 * n_rep, 32)
    k = torch.randn(1, heads, 16, 32)
    v = torch.randn(1, heads, 16 * n_rep, 32)
    info = run_pass(M(), (q, k, v))

    assert info["ri_before"] == 1
    # Key operand still routes through repeat_interleave (not rewired), no enable_gqa.
    assert not _kv_operands_are_rewired(info["gm"])
    assert _enable_gqa_flags(info["gm"]) == [None]


@pytest.mark.push
@pytest.mark.nightly
@pytest.mark.single_device
@pytest.mark.record_test_properties(category=Category.GRAPH_TEST)
def test_nonuniform_repeat_interleave_not_erased():
    """A per-head `repeats` tensor does not match enable_gqa's uniform interleave.

    repeats=[3, 1] along the head axis inflates 2 kv heads to 4 (== q_heads), so the
    output-shape gate alone would accept it -- but the head ordering differs from
    enable_gqa, so the pass must refuse based on `repeats` not being a scalar int.
    """
    kv_heads = 2
    q_heads = 4  # 3 + 1

    class M(nn.Module):
        def forward(self, q, k, v):
            repeats = torch.tensor([3, 1])
            return SDPA(
                q,
                k.repeat_interleave(repeats, dim=-3),
                v.repeat_interleave(repeats, dim=-3),
            )

    q, k, v = _qkv(1, q_heads, kv_heads, 16, 32)
    info = run_pass(M(), (q, k, v))

    # K/V stay inflated and enable_gqa is not set.
    assert _kv_head_counts(info["gm"]) == [(q_heads, q_heads)]
    assert _enable_gqa_flags(info["gm"]) == [None]


@pytest.mark.push
@pytest.mark.nightly
@pytest.mark.single_device
@pytest.mark.record_test_properties(category=Category.GRAPH_TEST)
def test_one_sided_expansion_not_erased():
    """Only key is repeat_kv-expanded; value already has q_heads.

    A one-sided rewrite would leave SDPA with a de-inflated key + full-head value
    under enable_gqa (mismatched K/V heads), so the pass must skip the node entirely.
    """
    kv_heads, n_rep = 2, 4
    q_heads = kv_heads * n_rep

    class M(nn.Module):
        def forward(self, q, k, v):
            # key is expanded from kv_heads -> q_heads; value is already q_heads.
            return SDPA(q, repeat_kv_hf(k, n_rep), v)

    q = torch.randn(1, q_heads, 16, 32)
    k = torch.randn(1, kv_heads, 16, 32)
    v = torch.randn(1, q_heads, 16, 32)
    info = run_pass(M(), (q, k, v))

    # Nothing rewired: key stays inflated, no enable_gqa.
    assert _kv_head_counts(info["gm"]) == [(q_heads, q_heads)]
    assert _enable_gqa_flags(info["gm"]) == [None]


@pytest.mark.push
@pytest.mark.nightly
@pytest.mark.single_device
@pytest.mark.record_test_properties(category=Category.GRAPH_TEST)
def test_pass_is_noop_without_sdpa():
    class M(nn.Module):
        def forward(self, x):
            return x @ x.transpose(-1, -2)

    x = torch.randn(1, 4, 8, 8)
    info = run_pass(M(), (x,))
    assert _sdpa_nodes(info["gm"]) == []


# ===========================================================================
# Direct unit tests for small helpers
# ===========================================================================
@pytest.mark.push
@pytest.mark.nightly
@pytest.mark.single_device
@pytest.mark.record_test_properties(category=Category.GRAPH_TEST)
def test_normalize_dim():
    assert _normalize_dim(-3, 4) == 1
    assert _normalize_dim(-1, 4) == 3
    assert _normalize_dim(1, 4) == 1
    assert _normalize_dim(0, 4) == 0


@pytest.mark.push
@pytest.mark.nightly
@pytest.mark.single_device
@pytest.mark.record_test_properties(category=Category.GRAPH_TEST)
def test_sdpa_operand_positional_and_keyword():
    """_sdpa_operand resolves operands whether positional or keyword."""
    torch._dynamo.reset()
    captured = {}

    class M(nn.Module):
        def forward(self, q, k, v):
            # value passed by keyword, key positional
            return SDPA(q, k, value=v)

    def backend(gm, example_inputs):
        captured["gm"] = gm
        return gm.forward

    q, k, v = _qkv(1, 4, 4, 8, 16)
    torch.compile(M(), backend=backend, fullgraph=True)(q, k, v)

    (sn,) = _sdpa_nodes(captured["gm"])
    # query + key positional; value keyword.
    assert _fx_node_shape(_sdpa_operand(sn, 0, "query")) == (1, 4, 8, 16)
    assert _fx_node_shape(_sdpa_operand(sn, 1, "key")) == (1, 4, 8, 16)
    assert _fx_node_shape(_sdpa_operand(sn, 2, "value")) == (1, 4, 8, 16)
