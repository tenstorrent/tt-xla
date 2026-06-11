# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""CPU unit tests for DeepSeek Sparse Attention (DSA) on TT.

These exercise the indexer + sparse-MLA path in ``vllm_tt.attention_dsa`` purely
on CPU (the tt:: custom ops have CPU fallbacks), so no TT hardware is required.

Coverage:
  * The lightning indexer's top-k selection (``compute_dsa_sparse_mask``) matches
    an independent reference for both prefill and decode, including causal
    masking and the indexer K cache write/update.
  * ``TTDSAAttentionBackendImpl.forward`` reuses ``flash_mla_prefill`` /
    ``paged_flash_mla_decode`` correctly: its output matches dense MLA attention
    evaluated with the same sparse mask, and the latent KV cache is filled.
  * Sparsity actually changes the result (DSA != dense MLA when topk < seq_len).

Tiny dims + float32 are used for fast, exact comparison; the logic is dim-agnostic.
"""

import math
from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn
import tt_torch.custom_ops  # noqa: F401  registers torch.ops.tt.*
from vllm_tt.attention import TTMetadata
from vllm_tt.attention_dsa import TTDSAAttentionBackendImpl, compute_dsa_sparse_mask

BLOCK_SIZE = 32  # TTMLAAttentionBackend.get_page_size

# Small but representative dims (DeepSeek-V3.2 shapes scaled way down).
N = 4  # attention heads
P = 8  # qk_nope_head_dim
R = 8  # qk_rope_head_dim
L = 16  # kv_lora_rank
V = 8  # v_head_dim
Q_LORA = 24  # q_lora_rank
HIDDEN = 40
HEAD_DIM = L + R  # latent kv head dim

# Indexer dims
IDX_NH = 4  # index_n_heads
IDX_HD = 16  # index_head_dim
IDX_RD = R  # indexer rope dim (== qk_rope_head_dim)


def _identity_rope(positions, q, k):
    """Stand-in rope: shape-preserving identity (the real rope is exercised by
    the MLA path already; here we only validate indexer selection + plumbing)."""
    return q, k


class _StubIndexer(nn.Module):
    """Minimal stand-in for vllm...deepseek_v2.Indexer (weights + config only).

    ``compute_dsa_sparse_mask`` reads exactly these submodules/attributes; the
    ``k_cache.kv_cache`` tensor is the bound indexer cache.
    """

    def __init__(self, topk_tokens: int):
        super().__init__()
        self.wq_b = nn.Linear(Q_LORA, IDX_NH * IDX_HD, bias=False)
        self.wk = nn.Linear(HIDDEN, IDX_HD, bias=False)
        self.k_norm = nn.LayerNorm(IDX_HD, eps=1e-6)
        self.weights_proj = nn.Linear(HIDDEN, IDX_NH, bias=False)
        self.n_head = IDX_NH
        self.head_dim = IDX_HD
        self.rope_dim = IDX_RD
        self.softmax_scale = IDX_HD**-0.5
        self.topk_tokens = topk_tokens
        self.k_cache = SimpleNamespace(kv_cache=torch.tensor([]))


def _make_indexer(topk_tokens: int) -> _StubIndexer:
    torch.manual_seed(1234)
    idx = _StubIndexer(topk_tokens).to(torch.float32)
    # Non-trivial weights so the indexer scores discriminate between tokens.
    for m in (idx.wq_b, idx.wk, idx.weights_proj):
        nn.init.normal_(m.weight, std=0.5)
    return idx


def _mla_scale() -> float:
    return (P + R) ** -0.5


def _absorbed_weights():
    """Random MLA absorbed weights W_UK_T [N,P,L], W_UV [N,L,V] (cf. MLA test)."""
    weight = torch.randn(N * (P + V), L, dtype=torch.float32) / math.sqrt(L)
    kv_b = weight.t().contiguous().view(L, N, P + V)
    W_UK, W_UV = kv_b.split([P, V], dim=-1)
    W_UV = W_UV.transpose(0, 1).contiguous()  # [N, L, V]
    W_UK_T = W_UK.permute(1, 2, 0).contiguous()  # [N, P, L]
    return W_UK_T, W_UV


def _make_impl() -> TTDSAAttentionBackendImpl:
    return TTDSAAttentionBackendImpl(
        num_heads=N,
        head_size=HEAD_DIM,
        scale=_mla_scale(),
        num_kv_heads=1,
        alibi_slopes=None,
        sliding_window=None,
        kv_cache_dtype="auto",
        logits_soft_cap=None,
        attn_type="decoder",
        kv_sharing_target_layer_name=None,
        q_lora_rank=Q_LORA,
        kv_lora_rank=L,
        qk_nope_head_dim=P,
        qk_rope_head_dim=R,
        qk_head_dim=P + R,
        v_head_dim=V,
    )


# --------------------------------------------------------------------------- #
# Independent reference computations
# --------------------------------------------------------------------------- #
def _ref_index_scores(indexer, hidden_states, qr, positions, users, S):
    """Independent reimplementation of the lightning-indexer score matrix.

    Uses a different einsum factorization than ``attention_dsa._index_scores`` so
    it is a genuine cross-check of q/k projection + ReLU-weighted scoring.
    Returns [users, S, Sk] where Sk == S (prefill: keys are the same tokens).
    """
    NH, HD, RD = indexer.n_head, indexer.head_dim, indexer.rope_dim
    q = indexer.wq_b(qr).view(-1, NH, HD)
    q_pe, q_nope = q[..., :RD], q[..., RD:]
    k = indexer.k_norm(indexer.wk(hidden_states))
    k_pe, k_nope = k[..., :RD], k[..., RD:]
    q_pe, k_pe = _identity_rope(positions, q_pe, k_pe.unsqueeze(1))
    q = torch.cat([q_pe.reshape(-1, NH, RD), q_nope], dim=-1).view(users, S, NH, HD)
    k = torch.cat([k_pe.reshape(-1, RD), k_nope], dim=-1).view(users, S, HD)
    w = (
        indexer.weights_proj(hidden_states) * (indexer.softmax_scale * NH**-0.5)
    ).view(users, S, NH)
    dots = torch.relu(torch.einsum("uihd,ujd->uijh", q, k))  # [u, Sq, Sk, NH]
    return torch.einsum("uijh,uih->uij", dots, w)  # [u, Sq, Sk]


def _ref_indexer_keys(indexer, hidden_states, positions):
    """The indexer K the cache should hold: [T, HD].

    With identity rope, ``cat([rope(k_pe), k_nope]) == k_norm(wk(x))``.
    """
    return indexer.k_norm(indexer.wk(hidden_states))


def _keep_from_mask(mask_2d: torch.Tensor) -> torch.Tensor:
    """Bool 'kept' map (additive mask ~0 -> kept, large-neg -> dropped)."""
    return mask_2d > (torch.finfo(mask_2d.dtype).min / 2)


def _ref_dsa_prefill_out(q_nope, q_pe, kv_c, k_pe, W_UK_T, W_UV, mask, users, S):
    scale = _mla_scale()
    q_nope = q_nope.view(users, S, N, P)
    q_pe = q_pe.view(users, S, N, R)
    kv_c = kv_c.view(users, S, L)
    k_pe = k_pe.view(users, S, 1, R)
    q_nope_lat = torch.einsum("bsnp,npl->bsnl", q_nope, W_UK_T)
    q_lat = torch.cat([q_nope_lat, q_pe], dim=-1)  # [b,S,N,L+R]
    k_lat = torch.cat([kv_c.unsqueeze(2), k_pe], dim=-1).squeeze(2)  # [b,S,L+R]
    sc = torch.einsum("bsnd,btd->bsnt", q_lat, k_lat) * scale  # [b,Sq,N,Sk]
    sc = sc + mask.permute(0, 2, 1, 3)  # mask [b,1,Sq,Sk] -> [b,Sq,1,Sk]
    attn = torch.softmax(sc, dim=-1)
    v_lat = k_lat[..., :L]  # [b,Sk,L]
    out_lat = torch.einsum("bsnt,btl->bsnl", attn, v_lat)  # [b,Sq,N,L]
    out = torch.einsum("bsnl,nlv->bsnv", out_lat, W_UV)  # [b,Sq,N,V]
    return out.reshape(users * S, N * V)


def _gather(cache, page_table, num_positions):
    """Read logical positions [0, num_positions) out of a paged cache."""
    users = page_table.shape[0]
    out = torch.zeros(users, num_positions, cache.shape[-1], dtype=cache.dtype)
    for u in range(users):
        for p in range(num_positions):
            blk = int(page_table[u, p // BLOCK_SIZE])
            out[u, p] = cache[blk, 0, p % BLOCK_SIZE, :]
    return out


def _pcc(a: torch.Tensor, b: torch.Tensor) -> float:
    x, y = a.flatten().float(), b.flatten().float()
    if torch.allclose(x, y, rtol=1e-3, atol=1e-3):
        return 1.0
    vx, vy = x - x.mean(), y - y.mean()
    denom = vx.norm() * vy.norm()
    return 1.0 if denom == 0 else float((vx @ vy) / denom)


# --------------------------------------------------------------------------- #
# Prefill
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("users,S,topk", [(1, 32, 8), (2, 32, 8), (1, 64, 16)])
def test_dsa_prefill(users, S, topk):
    torch.manual_seed(0)
    tokens = users * S
    blocks_per_user = (S + BLOCK_SIZE - 1) // BLOCK_SIZE
    num_blocks = users * blocks_per_user
    page_table = torch.arange(num_blocks, dtype=torch.int32).view(
        users, blocks_per_user
    )
    cache_position = torch.full((users,), S - 1, dtype=torch.int32)
    metadata = TTMetadata(
        cache_position=cache_position,
        attn_mask=None,
        page_table=page_table,
        is_causal=True,
        fill_page_table=page_table,
    )

    # Indexer inputs.
    indexer = _make_indexer(topk)
    hidden_states = torch.randn(tokens, HIDDEN)
    qr = torch.randn(tokens, Q_LORA)
    positions = torch.arange(tokens, dtype=torch.int32)
    indexer_cache = torch.zeros(num_blocks, 1, BLOCK_SIZE, IDX_HD)

    # MLA inputs (independent of the indexer inputs for this unit test).
    q_nope = torch.randn(tokens, N, P)
    q_pe = torch.randn(tokens, N, R)
    kv_c = torch.randn(tokens, L)
    k_pe = torch.randn(tokens, 1, R)
    latent_cache = torch.zeros(num_blocks, 1, BLOCK_SIZE, HEAD_DIM)
    W_UK_T, W_UV = _absorbed_weights()
    layer = SimpleNamespace(W_UK_T=W_UK_T, W_UV=W_UV)
    impl = _make_impl()

    # ---- DSA mask (mutates indexer_cache) ----
    mask = compute_dsa_sparse_mask(
        indexer,
        _identity_rope,
        indexer_cache,
        hidden_states,
        qr,
        positions,
        metadata,
        is_prefill=True,
        users=users,
        seq_len=S,
    )
    assert mask.shape == (users, 1, S, S)

    # ---- (1) Indexer top-k selection matches the independent reference ----
    ref_scores = _ref_index_scores(indexer, hidden_states, qr, positions, users, S)
    neg = torch.finfo(ref_scores.dtype).min
    causal = torch.triu(torch.full((S, S), neg), diagonal=1)
    kth = min(topk, S)
    ref_idx = (ref_scores + causal).topk(kth, dim=-1).indices
    ref_sel = torch.zeros(users, S, S, dtype=torch.bool).scatter_(-1, ref_idx, True)
    allowed = torch.tril(torch.ones(S, S, dtype=torch.bool))  # j <= i
    ref_keep = ref_sel & allowed.unsqueeze(0)

    impl_keep = _keep_from_mask(mask.squeeze(1))
    assert torch.equal(impl_keep, ref_keep), "indexer top-k selection mismatch"
    # Each query position keeps at most topk and at least 1 (itself).
    kept_per_row = impl_keep.sum(-1)
    assert (kept_per_row >= 1).all() and (kept_per_row <= kth).all()

    # ---- (2) Indexer K cache was filled with the projected keys ----
    expect_k = _ref_indexer_keys(indexer, hidden_states, positions).view(
        users, S, IDX_HD
    )
    got_k = _gather(indexer_cache, page_table, S)
    assert torch.allclose(got_k, expect_k, atol=1e-4), "indexer K cache mis-filled"

    # ---- (3) Forward output matches dense MLA with the same sparse mask ----
    out = impl.forward(
        q=(q_nope, q_pe),
        kv_c_normed=kv_c,
        k_pe=k_pe,
        kv_cache=latent_cache,
        attn_metadata=metadata,
        layer=layer,
        output=None,
        attn_mask=mask,
    )
    ref_out = _ref_dsa_prefill_out(
        q_nope, q_pe, kv_c, k_pe, W_UK_T, W_UV, mask, users, S
    )
    assert out.shape == ref_out.shape == (tokens, N * V)
    assert _pcc(out, ref_out) >= 0.999, "DSA prefill output != dense MLA w/ mask"

    # ---- (4) Latent KV cache filled with the latent K ----
    expect_lat = torch.cat([kv_c, k_pe.squeeze(1)], dim=-1).view(users, S, HEAD_DIM)
    got_lat = _gather(latent_cache, page_table, S)
    assert torch.allclose(got_lat, expect_lat, atol=1e-4), "latent cache mis-filled"

    # ---- (5) Sparsity actually matters: DSA differs from full dense MLA ----
    if kth < S:
        full_mask = torch.zeros_like(mask)  # keep everything except future
        causal4 = torch.triu(
            torch.full((S, S), torch.finfo(mask.dtype).min), diagonal=1
        )
        full_mask = full_mask + causal4.view(1, 1, S, S)
        dense_out = _ref_dsa_prefill_out(
            q_nope, q_pe, kv_c, k_pe, W_UK_T, W_UV, full_mask, users, S
        )
        assert _pcc(out, dense_out) < 0.999, "sparse mask had no effect vs dense MLA"


# --------------------------------------------------------------------------- #
# Decode
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("users,cur_pos,topk", [(1, 40, 8), (2, 48, 8), (1, 30, 4)])
def test_dsa_decode(users, cur_pos, topk):
    torch.manual_seed(0)
    blocks_per_user = cur_pos // BLOCK_SIZE + 1
    num_blocks = users * blocks_per_user
    max_seq = blocks_per_user * BLOCK_SIZE
    page_table = torch.arange(num_blocks, dtype=torch.int32).view(
        users, blocks_per_user
    )
    cache_position = torch.full((users,), cur_pos, dtype=torch.int32)
    metadata = TTMetadata(
        cache_position=cache_position,
        attn_mask=None,
        page_table=page_table,
        is_causal=True,
        fill_page_table=page_table,
    )

    # One decode token per user.
    indexer = _make_indexer(topk)
    hidden_states = torch.randn(users, HIDDEN)
    qr = torch.randn(users, Q_LORA)
    positions = cache_position.clone()
    # Pre-seed indexer cache with random history (only [0, cur_pos) is "valid").
    indexer_cache = torch.randn(num_blocks, 1, BLOCK_SIZE, IDX_HD)
    indexer_cache_before = indexer_cache.clone()

    q_nope = torch.randn(users, N, P)
    q_pe = torch.randn(users, N, R)
    kv_c = torch.randn(users, L)
    k_pe = torch.randn(users, 1, R)
    latent_cache = torch.randn(num_blocks, 1, BLOCK_SIZE, HEAD_DIM)
    latent_cache_before = latent_cache.clone()
    W_UK_T, W_UV = _absorbed_weights()
    layer = SimpleNamespace(W_UK_T=W_UK_T, W_UV=W_UV)
    impl = _make_impl()

    mask = compute_dsa_sparse_mask(
        indexer,
        _identity_rope,
        indexer_cache,
        hidden_states,
        qr,
        positions,
        metadata,
        is_prefill=False,
        users=users,
        seq_len=1,
    )
    assert mask.shape == (users, 1, 1, max_seq)

    # ---- (1) Indexer cache updated at cur_pos; selection matches reference ----
    new_k = _ref_indexer_keys(indexer, hidden_states, positions)  # [users, IDX_HD]
    # Build the reference k history: pre-seed gathered, with new_k written at cur_pos.
    k_hist = _gather(indexer_cache_before, page_table, max_seq)  # [users, max_seq, HD]
    for u in range(users):
        k_hist[u, cur_pos] = new_k[u]
    # Confirm the op wrote the new token where expected.
    got_hist = _gather(indexer_cache, page_table, max_seq)
    assert torch.allclose(
        got_hist[:, : cur_pos + 1], k_hist[:, : cur_pos + 1], atol=1e-4
    )

    # Independent indexer scores over the full history.
    NH, HD, RD = indexer.n_head, indexer.head_dim, indexer.rope_dim
    q = indexer.wq_b(qr).view(users, NH, HD)
    q_pe_i, q_nope_i = q[..., :RD], q[..., RD:]
    q_idx = torch.cat([q_pe_i, q_nope_i], dim=-1)  # identity rope
    w = indexer.weights_proj(hidden_states) * (indexer.softmax_scale * NH**-0.5)
    dots = torch.relu(torch.einsum("unh,uth->utn", q_idx, k_hist))  # [u, max_seq, NH]
    ref_scores = torch.einsum("utn,un->ut", dots, w)  # [u, max_seq]
    neg = torch.finfo(ref_scores.dtype).min
    pos_range = torch.arange(max_seq)
    future = pos_range.view(1, max_seq) > cache_position.view(users, 1)
    ref_scores = ref_scores.masked_fill(future, neg)
    kth = min(topk, max_seq)
    ref_idx = ref_scores.topk(kth, dim=-1).indices
    ref_sel = torch.zeros(users, max_seq, dtype=torch.bool).scatter_(-1, ref_idx, True)
    ref_keep = ref_sel & ~future
    impl_keep = _keep_from_mask(mask.view(users, max_seq))
    assert torch.equal(impl_keep, ref_keep), "decode indexer top-k mismatch"

    # ---- (2) Decode output matches dense MLA decode with the same mask ----
    out = impl.forward(
        q=(q_nope, q_pe),
        kv_c_normed=kv_c,
        k_pe=k_pe,
        kv_cache=latent_cache,
        attn_metadata=metadata,
        layer=layer,
        output=None,
        attn_mask=mask,
    )

    # Reference: update latent cache at cur_pos, gather, dense attention w/ mask.
    new_lat = torch.cat([kv_c, k_pe.squeeze(1)], dim=-1)  # [users, L+R]
    lat_hist = _gather(latent_cache_before, page_table, max_seq)
    for u in range(users):
        lat_hist[u, cur_pos] = new_lat[u]
    q_nope_lat = torch.einsum("bnp,npl->bnl", q_nope, W_UK_T)
    q_lat = torch.cat([q_nope_lat, q_pe], dim=-1)  # [users, N, L+R]
    sc = torch.einsum("bnd,btd->bnt", q_lat, lat_hist) * _mla_scale()  # [u, N, max_seq]
    sc = sc + mask.view(users, 1, max_seq)
    attn = torch.softmax(sc, dim=-1)
    v_all = lat_hist[..., :L]
    out_lat = torch.einsum("bnt,btl->bnl", attn, v_all)
    ref_out = torch.einsum("bnl,nlv->bnv", out_lat, W_UV).reshape(users, N * V)

    assert out.shape == ref_out.shape == (users, N * V)
    assert _pcc(out, ref_out) >= 0.999, "DSA decode output != dense MLA decode w/ mask"


def test_dsa_decode_without_cache_falls_back():
    """Decode before the indexer cache is bound returns None (plain causal MLA)."""
    torch.manual_seed(0)
    users = 1
    metadata = TTMetadata(
        cache_position=torch.zeros(users, dtype=torch.int32),
        attn_mask=None,
        page_table=torch.zeros(users, 1, dtype=torch.int32),
        is_causal=True,
        fill_page_table=torch.zeros(users, 1, dtype=torch.int32),
    )
    indexer = _make_indexer(8)
    mask = compute_dsa_sparse_mask(
        indexer,
        _identity_rope,
        None,  # unbound indexer cache
        torch.randn(users, HIDDEN),
        torch.randn(users, Q_LORA),
        torch.zeros(users, dtype=torch.int32),
        metadata,
        is_prefill=False,
        users=users,
        seq_len=1,
    )
    assert mask is None
