# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""DeepSeek Sparse Attention paths in the TT MLA backend impl.

Drives ``TTMLAAttentionBackendImpl`` directly with a stubbed indexer, so the
sparse prefill branch, the masked decode branch, and the dense-fallback
predicates are all exercised without loading a model. Shared scaffolding is in
``conftest.py``; the plain dense-MLA equivalents live in
``test_mla_attention_impl.py``.
"""

from types import SimpleNamespace

import pytest
import torch
import torch_xla
import torch_xla.runtime as xr

# Registers the tt:: custom ops.
import tt_torch.custom_ops  # noqa: F401
from conftest import (
    BLOCK_SIZE,
    REQUIRED_PCC,
    gather_cache,
    latent_k,
    maybe_mesh,
    pcc,
    run_mla_impl,
)
from tt_torch.custom_ops import TOPK_LARGE_INDICES_SENTINEL

from tests.utils import parametrize_arch


def _indexer_stub(topk_indices, topk_tokens, k_chunk_size=32):
    """Minimal stand-in for TTIndexer: the impl only reads these three attrs."""
    return SimpleNamespace(
        topk_indices=topk_indices,
        topk_tokens=topk_tokens,
        k_chunk_size=k_chunk_size,
    )


def _causal_topk_indices(users, seq_len, topk, key_seq_len=None, device="cpu"):
    """Indices naming each row's causally visible keys, sentinel-padded.

    Row ``s`` sees keys ``[0, s]``; slots beyond that are invalid. Mirrors exactly
    what ``TTIndexer`` publishes (per-user, ``[users, 1, s, topk]``).
    """
    key_seq_len = key_seq_len or seq_len
    idx = torch.full(
        (users, 1, seq_len, topk), TOPK_LARGE_INDICES_SENTINEL, dtype=torch.int64
    )
    for u in range(users):
        for s in range(seq_len):
            visible = min(s + 1, topk)
            idx[u, 0, s, :visible] = torch.arange(visible)
    return idx.to(torch.uint32).to(device)


def _decode_topk_indices(users, cur_pos, topk, device="cpu"):
    """Decode-shaped indices ([users, 1, 1, topk]) over keys [0, cur_pos]."""
    idx = torch.full(
        (users, 1, 1, topk), TOPK_LARGE_INDICES_SENTINEL, dtype=torch.int64
    )
    visible = min(cur_pos + 1, topk)
    for u in range(users):
        idx[u, 0, 0, :visible] = torch.arange(visible)
    return idx.to(torch.uint32).to(device)


def _prefill_inputs(cfg, dtype, users, seq_len, blocks_per_user):
    N = cfg["num_attention_heads"]
    L = cfg["kv_lora_rank"]
    R = cfg["qk_rope_head_dim"]
    P = cfg["qk_nope_head_dim"]
    tokens = users * seq_len
    return (
        torch.randn(tokens, N, P, dtype=dtype),
        torch.randn(tokens, N, R, dtype=dtype),
        torch.randn(tokens, L, dtype=dtype),
        torch.randn(tokens, 1, R, dtype=dtype),
        torch.zeros(users * blocks_per_user, 1, BLOCK_SIZE, L + R, dtype=dtype),
        torch.full((users,), seq_len - 1, dtype=torch.int32),
        torch.arange(users * blocks_per_user, dtype=torch.int32).view(
            users, blocks_per_user
        ),
    )


# --------------------------------------------------------------------------- #
# Sparse prefill
# --------------------------------------------------------------------------- #
@pytest.mark.nightly
@parametrize_arch(["llmbox"])
@pytest.mark.parametrize(
    "users, seq_len, topk", [(1, 128, 128), (2, 128, 128), (1, 256, 128)]
)
def test_dsa_prefill_impl(users, seq_len, topk, arch, deepseek_v32_mla):
    """Sparse prefill (tt.sparse_sdpa) on device vs the same impl on CPU."""
    xr.set_device_type("TT")
    torch.manual_seed(0)

    params = deepseek_v32_mla
    cfg = params["cfg"]
    dtype = params["act_dtype"]
    N, V = cfg["num_attention_heads"], cfg["v_head_dim"]
    mesh = maybe_mesh(arch, N)

    blocks_per_user = (seq_len + BLOCK_SIZE - 1) // BLOCK_SIZE
    inputs = _prefill_inputs(cfg, dtype, users, seq_len, blocks_per_user)
    indices = _causal_topk_indices(users, seq_len, topk)

    golden, _ = run_mla_impl(
        torch.device("cpu"),
        params,
        inputs,
        indexer=_indexer_stub(indices, topk),
    )
    device = torch_xla.device()
    device_out, _ = run_mla_impl(
        device,
        params,
        inputs,
        mesh=mesh,
        indexer=_indexer_stub(indices.to(device), topk),
    )
    torch_xla.sync()
    device_out = device_out.cpu()

    assert device_out.shape == golden.shape == (users * seq_len, N * V)
    got = pcc(device_out, golden)
    assert got >= REQUIRED_PCC, f"DSA prefill PCC {got:.5f} < {REQUIRED_PCC}"


@pytest.mark.nightly
@parametrize_arch(["llmbox"])
@pytest.mark.parametrize("users, seq_len", [(1, 128), (2, 64)])
def test_dsa_prefill_equals_dense_when_topk_covers_seq(
    users, seq_len, arch, deepseek_v32_mla
):
    """Sparse prefill with topk >= seq_len must equal the dense flash-MLA path.

    This is the regression guard for ``dsa_prefill_uses_sparse``: when top-k covers
    every causally visible key, the sparse and dense kernels compute the same
    thing, so the predicate is free to pick either.
    """
    xr.set_device_type("TT")
    torch.manual_seed(0)

    params = deepseek_v32_mla
    cfg = params["cfg"]
    dtype = params["act_dtype"]
    mesh = maybe_mesh(arch, cfg["num_attention_heads"])

    blocks_per_user = (seq_len + BLOCK_SIZE - 1) // BLOCK_SIZE
    inputs = _prefill_inputs(cfg, dtype, users, seq_len, blocks_per_user)
    device = torch_xla.device()

    # topk == seq_len: every visible key is selected.
    indices = _causal_topk_indices(users, seq_len, seq_len).to(device)
    sparse_out, sparse_cache = run_mla_impl(
        device, params, inputs, mesh=mesh, indexer=_indexer_stub(indices, seq_len)
    )
    # indexer=None -> the impl takes the dense tt.flash_mla_prefill branch.
    dense_out, dense_cache = run_mla_impl(device, params, inputs, mesh=mesh)
    torch_xla.sync()
    sparse_out, dense_out = sparse_out.cpu(), dense_out.cpu()

    got = pcc(sparse_out, dense_out)
    assert got >= REQUIRED_PCC, (
        f"sparse vs dense prefill PCC {got:.5f} < {REQUIRED_PCC}; with "
        "topk >= seq_len the two must agree."
    )
    # Both paths must persist the same latent KV.
    assert pcc(sparse_cache.cpu(), dense_cache.cpu()) >= REQUIRED_PCC


@pytest.mark.nightly
@parametrize_arch(["llmbox"])
def test_dsa_prefill_fills_latent_cache(arch, deepseek_v32_mla):
    """The sparse branch must still persist the latent KV for later decodes."""
    xr.set_device_type("TT")
    torch.manual_seed(0)

    params = deepseek_v32_mla
    cfg = params["cfg"]
    dtype = params["act_dtype"]
    users, seq_len, topk = 1, 128, 128
    L, R = cfg["kv_lora_rank"], cfg["qk_rope_head_dim"]
    mesh = maybe_mesh(arch, cfg["num_attention_heads"])

    blocks_per_user = seq_len // BLOCK_SIZE
    inputs = _prefill_inputs(cfg, dtype, users, seq_len, blocks_per_user)
    _, _, kv_c, k_pe, _, _, page_table = inputs
    device = torch_xla.device()

    _, cache = run_mla_impl(
        device,
        params,
        inputs,
        mesh=mesh,
        indexer=_indexer_stub(
            _causal_topk_indices(users, seq_len, topk).to(device), topk
        ),
    )
    torch_xla.sync()

    expected = latent_k(kv_c, k_pe).view(users, seq_len, L + R)
    assert torch.allclose(
        gather_cache(cache.cpu(), page_table, seq_len), expected
    ), "sparse prefill did not fill the latent cache correctly"


# --------------------------------------------------------------------------- #
# Decode
# --------------------------------------------------------------------------- #
@pytest.mark.push
@pytest.mark.single_device
def test_paged_mla_decode_rejects_non_causal_on_device():
    """A non-causal paged MLA decode must fail loudly, not abort at runtime.

    ``ttnn::prim::sdpa_decode`` asserts ``is_causal``
    (``sdpa_decode_device_operation.cpp:28``, "Multi-latent attention decode only
    tested for causal!"), so an additive-mask decode aborts the process with a
    TT_FATAL. That is why DSA decode gathers the cache and uses ``tt.sparse_sdpa``
    rather than masking the paged kernel; this pins the guard that turns the abort
    into an actionable Python error.
    """
    import torch_xla

    device = torch_xla.device()
    users, nqh, dh, v_dim, block_size, blocks = 1, 32, 576, 512, BLOCK_SIZE, 2
    max_seq_len = blocks * block_size

    query = torch.randn(1, users, nqh, dh, dtype=torch.bfloat16, device=device)
    cache = torch.randn(blocks, 1, block_size, dh, dtype=torch.bfloat16, device=device)
    page_table = torch.arange(blocks, dtype=torch.int32, device=device).view(
        users, blocks
    )
    mask = torch.zeros(users, 1, nqh, max_seq_len, dtype=torch.bfloat16, device=device)

    with pytest.raises(NotImplementedError, match="is_causal"):
        # `scale` is omitted rather than passed as None: these older MLA ops
        # annotate it `float = None`, which registers a non-nullable schema type.
        torch.ops.tt.paged_flash_mla_decode(
            query=query,
            key=cache,
            head_dim_v=v_dim,
            page_table=page_table,
            value=None,
            is_causal=False,
            attn_mask=mask,
        )


@pytest.mark.nightly
@parametrize_arch(["llmbox"])
@pytest.mark.parametrize("users, cur_pos", [(1, 95), (2, 63)])
def test_dsa_decode_sparse_equals_dense_when_all_visible(
    users, cur_pos, arch, deepseek_v32_mla
):
    """Sparse decode over every visible key == dense causal decode.

    This is the exactness proof that licenses the dense decode path whenever
    ``max_seq_len <= index_topk``: when top-k names every causally visible key, the
    gather + ``tt.sparse_sdpa`` path and the paged Flash MLA kernel must agree.
    It is also the only coverage of ``_forward_decode_sparse``.
    """
    xr.set_device_type("TT")
    torch.manual_seed(0)

    params = deepseek_v32_mla
    cfg = params["cfg"]
    dtype = params["act_dtype"]
    N = cfg["num_attention_heads"]
    L, R, P, V = (
        cfg["kv_lora_rank"],
        cfg["qk_rope_head_dim"],
        cfg["qk_nope_head_dim"],
        cfg["v_head_dim"],
    )
    mesh = maybe_mesh(arch, N)

    blocks_per_user = cur_pos // BLOCK_SIZE + 1
    num_blocks = users * blocks_per_user
    max_seq_len = blocks_per_user * BLOCK_SIZE

    inputs = (
        torch.randn(users, N, P, dtype=dtype),
        torch.randn(users, N, R, dtype=dtype),
        torch.randn(users, L, dtype=dtype),
        torch.randn(users, 1, R, dtype=dtype),
        torch.randn(num_blocks, 1, BLOCK_SIZE, L + R, dtype=dtype),
        torch.full((users,), cur_pos, dtype=torch.int32),
        torch.arange(num_blocks, dtype=torch.int32).view(users, blocks_per_user),
    )
    device = torch_xla.device()

    # topk covers max_seq_len, so every visible key is selected.
    topk = max_seq_len
    indices = _decode_topk_indices(users, cur_pos, topk).to(device)
    sparse_out, sparse_cache = run_mla_impl(
        device, params, inputs, mesh=mesh, indexer=_indexer_stub(indices, topk)
    )
    dense_out, dense_cache = run_mla_impl(device, params, inputs, mesh=mesh)
    torch_xla.sync()
    sparse_out, dense_out = sparse_out.cpu(), dense_out.cpu()

    assert sparse_out.shape == dense_out.shape == (users, N * V)
    got = pcc(sparse_out, dense_out)
    assert got >= REQUIRED_PCC, (
        f"sparse vs dense decode PCC {got:.5f} < {REQUIRED_PCC}; selecting every "
        "visible key must reproduce dense causal decode."
    )
    assert pcc(sparse_cache.cpu(), dense_cache.cpu()) >= REQUIRED_PCC


@pytest.mark.nightly
@parametrize_arch(["llmbox"])
def test_dsa_decode_preserves_cache(arch, deepseek_v32_mla):
    """A masked decode step must still append its token and keep the prefix."""
    xr.set_device_type("TT")
    torch.manual_seed(0)

    params = deepseek_v32_mla
    cfg = params["cfg"]
    dtype = params["act_dtype"]
    N = cfg["num_attention_heads"]
    L, R, P = cfg["kv_lora_rank"], cfg["qk_rope_head_dim"], cfg["qk_nope_head_dim"]
    users, cur_pos = 1, 95
    mesh = maybe_mesh(arch, N)

    blocks_per_user = cur_pos // BLOCK_SIZE + 1
    num_blocks = users * blocks_per_user
    max_seq_len = blocks_per_user * BLOCK_SIZE
    seeded = torch.randn(num_blocks, 1, BLOCK_SIZE, L + R, dtype=dtype)
    page_table = torch.arange(num_blocks, dtype=torch.int32).view(
        users, blocks_per_user
    )
    kv_c = torch.randn(users, L, dtype=dtype)
    k_pe = torch.randn(users, 1, R, dtype=dtype)

    inputs = (
        torch.randn(users, N, P, dtype=dtype),
        torch.randn(users, N, R, dtype=dtype),
        kv_c,
        k_pe,
        seeded,
        torch.full((users,), cur_pos, dtype=torch.int32),
        page_table,
    )

    _, cache = run_mla_impl(
        torch.device("cpu"),
        params,
        inputs,
        indexer=_indexer_stub(
            _decode_topk_indices(users, cur_pos, max_seq_len), max_seq_len
        ),
    )

    after = gather_cache(cache, page_table, cur_pos + 1)
    before = gather_cache(seeded, page_table, cur_pos + 1)
    assert torch.allclose(
        after[:, cur_pos, :], latent_k(kv_c, k_pe)
    ), "masked decode did not write the new token at cur_pos"
    assert torch.allclose(
        after[:, :cur_pos, :], before[:, :cur_pos, :]
    ), "masked decode clobbered prior context"
