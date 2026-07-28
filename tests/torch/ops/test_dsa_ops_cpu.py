# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""CPU checks for the DeepSeek Sparse Attention (DSA) ops.

Pins each op's CPU reference branch — which is the device tests' oracle — against
an independent implementation written from the op contracts, plus the invariants
that hold the three-op pipeline together:

    tt.indexer_score_dsa -> tt.topk_large_indices -> tt.sparse_sdpa

The load-bearing one is ``test_chain_matches_dense_causal_when_topk_covers_seq``:
it proves sparse attention over "all causally visible keys" *is* dense causal
attention, which is what licenses the dense fallback whenever ``S <= index_topk``.
"""

import pytest
import torch
import torch.nn.functional as F

# Registers the tt:: custom ops (including the three DSA ops).
import tt_torch.custom_ops  # noqa: F401
from tt_torch.custom_ops import (
    TOPK_LARGE_INDICES_SENTINEL,
    topk_large_indices_mask_invalid_slots,
)

pytestmark = pytest.mark.cpu

DTYPE = torch.bfloat16


# --------------------------------------------------------------------------- #
# Independent references
# --------------------------------------------------------------------------- #
def _reference_indexer_score(query, key, weights, chunk_start_idx):
    """Independent oracle: per-head relu(q.k) gated by `weights`, summed over
    heads, with future positions set to -inf."""
    b, hi, sq, _ = query.shape
    t = key.shape[2]
    out = torch.zeros(b, 1, sq, t, dtype=torch.float32)
    for bi in range(b):
        for s in range(sq):
            for ti in range(t):
                if ti > chunk_start_idx + s:
                    out[bi, 0, s, ti] = float("-inf")
                    continue
                acc = 0.0
                for h in range(hi):
                    dot = torch.dot(
                        query[bi, h, s].float(), key[bi, 0, ti].float()
                    ).clamp(min=0)
                    acc += float(dot) * float(weights[bi, h, s, 0])
                out[bi, 0, s, ti] = acc
    return out.to(query.dtype)


def _reference_sparse_sdpa(query, kv, indices, v_dim, scale):
    """Independent oracle: for each query row, softmax over only the key
    positions its `indices` row names, then read V from kv[..., :v_dim]."""
    b, h, s, dh = query.shape
    t = kv.shape[2]
    scale = dh**-0.5 if scale is None else scale
    out = torch.zeros(b, h, s, v_dim, dtype=torch.float32)
    for bi in range(b):
        for si in range(s):
            kept = sorted(
                {
                    int(v)
                    for v in indices[bi, 0, si].to(torch.int64).tolist()
                    if 0 <= int(v) < t
                }
            )
            assert kept, f"row {si} names no valid key; softmax would be NaN"
            keys = kv[bi, 0, kept].float()  # [n_kept, dh]
            for hi in range(h):
                logits = (keys @ query[bi, hi, si].float()) * scale
                probs = torch.softmax(logits, dim=-1)
                out[bi, hi, si] = probs @ keys[:, :v_dim]
    return out.to(query.dtype)


def _causal_score(b, hi, sq, t, d, chunk_start_idx=0, seed=0):
    g = torch.Generator().manual_seed(seed)
    q = torch.randn(b, hi, sq, d, dtype=DTYPE, generator=g)
    k = torch.randn(b, 1, t, d, dtype=DTYPE, generator=g)
    w = torch.randn(b, hi, sq, 1, dtype=DTYPE, generator=g)
    return q, k, w, torch.ops.tt.indexer_score_dsa(q, k, w, chunk_start_idx)


# --------------------------------------------------------------------------- #
# tt.indexer_score_dsa
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("chunk_start_idx", [0, 4, 16])
def test_indexer_score_matches_reference(chunk_start_idx):
    b, hi, sq, t, d = 1, 4, 8, 16, 32
    q, k, w, got = _causal_score(b, hi, sq, t, d, chunk_start_idx)
    want = _reference_indexer_score(q, k, w, chunk_start_idx)

    assert torch.equal(torch.isneginf(got), torch.isneginf(want))
    finite = torch.isfinite(want)
    torch.testing.assert_close(
        got[finite].float(), want[finite].float(), atol=2e-2, rtol=2e-2
    )


@pytest.mark.parametrize("chunk_start_idx", [0, 3, 8])
def test_indexer_score_causal_boundary_is_inclusive(chunk_start_idx):
    """Key ``t`` is visible to query ``s`` iff ``t <= chunk_start_idx + s``."""
    sq, t = 8, 24
    _, _, _, score = _causal_score(1, 2, sq, t, 32, chunk_start_idx)
    for s in range(sq):
        visible = torch.isfinite(score[0, 0, s])
        expected = torch.arange(t) <= (chunk_start_idx + s)
        assert torch.equal(visible, expected), f"row {s} mask wrong"


def test_indexer_score_rejects_non_bfloat16():
    q = torch.randn(1, 2, 8, 32)
    k = torch.randn(1, 1, 8, 32)
    w = torch.randn(1, 2, 8, 1)
    with pytest.raises(Exception, match="bfloat16"):
        torch.ops.tt.indexer_score_dsa(q, k, w, 0)


# --------------------------------------------------------------------------- #
# tt.topk_large_indices
# --------------------------------------------------------------------------- #
def test_topk_matches_torch_topk_on_finite_input():
    g = torch.Generator().manual_seed(0)
    x = torch.randn(1, 1, 8, 256, dtype=DTYPE, generator=g)
    got = torch.ops.tt.topk_large_indices(x, 64)
    want = torch.topk(x.float(), k=64, dim=-1, largest=True, sorted=True).indices

    assert got.dtype == torch.uint32
    # Compare gathered values, not indices: bf16 ties make index order arbitrary.
    torch.testing.assert_close(
        torch.gather(x.float(), -1, got.to(torch.int64)),
        torch.gather(x.float(), -1, want),
    )


def test_topk_emits_sentinels_for_neginf_as_a_contiguous_tail():
    """The contract tt.sparse_sdpa depends on: masked slots are the sentinel, and
    they occupy the tail of every row."""
    sq, t, k = 32, 64, 16
    _, _, _, score = _causal_score(1, 4, sq, t, 32)
    idx = torch.ops.tt.topk_large_indices(score, k)
    signed = idx.to(torch.int32)

    # Row s has s+1 finite scores, so it needs max(0, k - (s+1)) sentinels.
    expected = (k - torch.arange(1, sq + 1)).clamp(min=0).to(torch.int32)
    assert torch.equal((signed < 0).sum(-1)[0, 0], expected)

    for s in range(sq):
        row = signed[0, 0, s].tolist()
        sentinel_slots = [i for i, v in enumerate(row) if v < 0]
        if sentinel_slots:
            assert sentinel_slots == list(
                range(sentinel_slots[0], k)
            ), f"row {s} sentinels are not a contiguous tail: {row}"
        # Every surviving index must be causally visible.
        assert all(v <= s for v in row if v >= 0), f"row {s} selected a future key"

    # Row 0 sees only key 0, so slots 1.. are all sentinel — check the raw value.
    assert int(idx[0, 0, 0, 0]) == 0
    assert int(idx[0, 0, 0, -1]) == TOPK_LARGE_INDICES_SENTINEL


@pytest.mark.parametrize(
    "k, match",
    [(0, "positive"), (8, r"\[16, 2048\]"), (24, "multiple of 16"), (4096, "2048")],
)
def test_topk_rejects_illegal_k(k, match):
    x = torch.randn(1, 1, 4, 4096, dtype=DTYPE)
    with pytest.raises(Exception, match=match):
        torch.ops.tt.topk_large_indices(x, k)


def test_topk_rejects_k_larger_than_row():
    x = torch.randn(1, 1, 4, 16, dtype=DTYPE)
    with pytest.raises(Exception, match="must be >= k"):
        torch.ops.tt.topk_large_indices(x, 32)


# --------------------------------------------------------------------------- #
# topk_large_indices_mask_invalid_slots (the decomposition-path repair)
# --------------------------------------------------------------------------- #
def test_mask_invalid_slots_repairs_decomposition_style_indices():
    """A plain topk (no sentinels) is what non-Blackhole lowering produces; the
    repair must invalidate exactly the non-visible slots and restore causality."""
    sq, t, k = 16, 64, 16
    _, _, _, score = _causal_score(1, 4, sq, t, 32)
    # Emulate the decomposition: ordinary indices everywhere, no sentinels.
    raw = torch.topk(score.float(), k=k, dim=-1, largest=True, sorted=True).indices
    raw = raw.to(torch.int64).to(torch.uint32)
    assert (raw.to(torch.int32) >= 0).all(), "precondition: no sentinels yet"

    visible = (torch.arange(sq, dtype=torch.int32) + 1).view(1, 1, sq, 1)
    fixed = topk_large_indices_mask_invalid_slots(raw, visible)

    assert fixed.dtype == torch.int32
    expected = (k - torch.arange(1, sq + 1)).clamp(min=0).to(torch.int32)
    assert torch.equal((fixed < 0).sum(-1)[0, 0], expected)
    for s in range(sq):
        row = fixed[0, 0, s].tolist()
        assert all(v <= s for v in row if v >= 0), f"row {s} kept a future key"
        invalid = [i for i, v in enumerate(row) if v < 0]
        if invalid:
            assert invalid == list(range(invalid[0], k)), f"row {s}: {row}"


def test_mask_invalid_slots_agrees_with_kernel_sentinels():
    """The repair must invalidate exactly the slots the kernel would sentinel, so
    the two architectures attend to the same key set."""
    sq, t, k = 32, 64, 16
    _, _, _, score = _causal_score(1, 4, sq, t, 32)
    # The op's CPU branch implements the kernel's contract.
    kernel_like = torch.ops.tt.topk_large_indices(score, k).to(torch.int32)
    visible = (torch.arange(sq, dtype=torch.int32) + 1).view(1, 1, sq, 1)
    repaired = topk_large_indices_mask_invalid_slots(
        torch.topk(score.float(), k=k, dim=-1, largest=True, sorted=True)
        .indices.to(torch.int64)
        .to(torch.uint32),
        visible,
    )
    assert torch.equal(kernel_like < 0, repaired < 0), "invalidated slots differ"


def test_mask_invalid_slots_output_is_consumable_by_sparse_sdpa():
    """int32 with -1 must mask exactly like a sentinel does: -1 matches no key
    position, so sparse_sdpa sees only the visible keys."""
    sq, t, k = 16, 32, 16
    dh, v_dim, h = 64, 32, 32
    _, _, _, score = _causal_score(1, 4, sq, t, 32)
    visible = (torch.arange(sq, dtype=torch.int32) + 1).view(1, 1, sq, 1)
    raw = (
        torch.topk(score.float(), k=k, dim=-1, largest=True, sorted=True)
        .indices.to(torch.int64)
        .to(torch.uint32)
    )
    repaired = topk_large_indices_mask_invalid_slots(raw, visible)

    g = torch.Generator().manual_seed(3)
    q = torch.randn(1, h, sq, dh, dtype=DTYPE, generator=g)
    kv = torch.randn(1, 1, t, dh, dtype=DTYPE, generator=g)

    got = torch.ops.tt.sparse_sdpa(q, kv, repaired, v_dim, None, 32)
    # Golden: the same op fed the kernel-contract (sentinel) indices.
    want = torch.ops.tt.sparse_sdpa(
        q, kv, torch.ops.tt.topk_large_indices(score, k), v_dim, None, 32
    )
    assert torch.isfinite(got.float()).all()
    torch.testing.assert_close(got.float(), want.float(), atol=2e-2, rtol=2e-2)


# --------------------------------------------------------------------------- #
# tt.sparse_sdpa
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("scale", [None, 0.125])
def test_sparse_sdpa_matches_reference(scale):
    b, h, s, t, dh, v_dim, topk = 1, 32, 8, 32, 64, 32, 16
    g = torch.Generator().manual_seed(0)
    q = torch.randn(b, h, s, dh, dtype=DTYPE, generator=g)
    kv = torch.randn(b, 1, t, dh, dtype=DTYPE, generator=g)
    # Distinct in-range indices per row, sorted, no sentinels.
    idx = torch.stack(
        [torch.randperm(t, generator=g)[:topk].sort().values for _ in range(s)]
    ).view(b, 1, s, topk)
    idx = idx.to(torch.int64).to(torch.uint32)

    got = torch.ops.tt.sparse_sdpa(q, kv, idx, v_dim, scale, 32)
    want = _reference_sparse_sdpa(q, kv, idx, v_dim, scale)
    assert got.shape == (b, h, s, v_dim)
    torch.testing.assert_close(got.float(), want.float(), atol=2e-2, rtol=2e-2)


def test_sparse_sdpa_ignores_sentinel_slots():
    """A sentinel tail must behave exactly like a shorter index list."""
    b, h, s, t, dh, v_dim, topk = 1, 32, 8, 32, 64, 32, 16
    g = torch.Generator().manual_seed(0)
    q = torch.randn(b, h, s, dh, dtype=DTYPE, generator=g)
    kv = torch.randn(b, 1, t, dh, dtype=DTYPE, generator=g)

    keep = 6
    base = torch.arange(topk).view(1, 1, 1, topk).expand(b, 1, s, topk).contiguous()
    padded = base.clone()
    padded[..., keep:] = TOPK_LARGE_INDICES_SENTINEL
    padded = padded.to(torch.int64).to(torch.uint32)

    got = torch.ops.tt.sparse_sdpa(q, kv, padded, v_dim, None, 32)
    # Reference over just the kept prefix, repeated to the same width.
    truncated = base[..., :keep].to(torch.int64).to(torch.uint32)
    want = _reference_sparse_sdpa(q, kv, truncated, v_dim, None)

    assert torch.isfinite(got.float()).all()
    torch.testing.assert_close(got.float(), want.float(), atol=2e-2, rtol=2e-2)


def test_sparse_sdpa_duplicate_indices_are_idempotent():
    """The mask is a set membership test, so naming a key twice must not
    double-weight it (matches the decomposition's `hitCount > 0`)."""
    b, h, s, t, dh, v_dim = 1, 32, 4, 32, 64, 32
    g = torch.Generator().manual_seed(0)
    q = torch.randn(b, h, s, dh, dtype=DTYPE, generator=g)
    kv = torch.randn(b, 1, t, dh, dtype=DTYPE, generator=g)

    unique = torch.arange(16).view(1, 1, 1, 16).expand(b, 1, s, 16).contiguous()
    dup = torch.cat([unique[..., :8], unique[..., :8], unique[..., 8:]], dim=-1)
    out_u = torch.ops.tt.sparse_sdpa(
        q, kv, unique.to(torch.int64).to(torch.uint32), v_dim, None, 32
    )
    out_d = torch.ops.tt.sparse_sdpa(
        q, kv, dup.to(torch.int64).to(torch.uint32), v_dim, None, 32
    )
    torch.testing.assert_close(out_u.float(), out_d.float(), atol=2e-2, rtol=2e-2)


def test_sparse_sdpa_rejects_v_dim_over_head_dim():
    q = torch.randn(1, 32, 4, 64, dtype=DTYPE)
    kv = torch.randn(1, 1, 32, 64, dtype=DTYPE)
    idx = torch.zeros(1, 1, 4, 16, dtype=torch.int64).to(torch.uint32)
    with pytest.raises(Exception, match="cannot exceed"):
        torch.ops.tt.sparse_sdpa(q, kv, idx, 128, None, 32)


# --------------------------------------------------------------------------- #
# The whole pipeline
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("seq_len", [32, 64, 128])
def test_chain_matches_dense_causal_when_topk_covers_seq(seq_len):
    """``sparse_sdpa(topk(indexer_score(...)))`` == dense causal SDPA whenever
    top-k can cover every causally visible key.

    Top-k over a row with at most ``topk`` finite (causally visible) scores keeps
    *every* visible key, so the sparse result is the dense causal one. This is the
    exactness argument behind ``dsa_prefill_uses_sparse`` / the dense decode path:
    below the top-k threshold the plugin may use the dense Flash MLA kernels
    without approximating anything.

    ``topk == seq_len`` is the only directly testable point of that regime: the op
    requires ``topk <= row width`` (== seq_len here), and coverage requires
    ``topk >= seq_len``. It is exactly the boundary the predicate switches on.
    """
    topk = seq_len
    h, dh, v_dim = 32, 576, 512
    _, _, _, score = _causal_score(1, 8, seq_len, seq_len, 128)
    idx = torch.ops.tt.topk_large_indices(score, topk)

    g = torch.Generator().manual_seed(1)
    q = torch.randn(1, h, seq_len, dh, dtype=DTYPE, generator=g)
    kv = torch.randn(1, 1, seq_len, dh, dtype=DTYPE, generator=g)

    sparse = torch.ops.tt.sparse_sdpa(q, kv, idx, v_dim, None, 32)
    dense = F.scaled_dot_product_attention(
        q.float(),
        kv.float(),
        kv[..., :v_dim].float(),
        is_causal=True,
        scale=dh**-0.5,
        enable_gqa=True,
    )
    assert torch.isfinite(sparse.float()).all()
    torch.testing.assert_close(sparse.float(), dense.float(), atol=3e-2, rtol=3e-2)


def test_chain_row_zero_always_has_a_valid_key():
    """Every row must name >= 1 valid key or sparse_sdpa's softmax is NaN. Row 0
    is the tightest case: it can only ever see key 0."""
    sq, topk = 32, 16
    _, _, _, score = _causal_score(1, 4, sq, sq, 32)
    idx = torch.ops.tt.topk_large_indices(score, topk).to(torch.int32)
    for s in range(sq):
        assert (idx[0, 0, s] >= 0).sum() >= 1, f"row {s} has no valid key"
    assert int(idx[0, 0, 0, 0]) == 0, "row 0 must select key 0"
