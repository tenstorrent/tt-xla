# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Dense MLA prefill / paged-decode tests for the TT OOT MLA backend impl.

Shared scaffolding (config, absorbed weights, ``run_mla_impl``, PCC, paged-cache
gather) lives in ``conftest.py``; the DSA variants live in
``test_dsa_prefill_impl.py``.
"""

import pytest
import torch
import torch_xla
import torch_xla.runtime as xr
from conftest import (
    BLOCK_SIZE,
    REQUIRED_PCC,
    gather_cache,
    latent_k,
    maybe_mesh,
    pcc,
    run_mla_impl,
)

from tests.utils import parametrize_arch


@pytest.mark.nightly
@parametrize_arch(["llmbox"])
@pytest.mark.parametrize("users, seq_len", [(1, 64), (2, 64), (1, 128)])
def test_mla_prefill_impl_deepseek_v3(users, seq_len, arch, deepseek_v3_mla):
    xr.set_device_type("TT")
    torch.manual_seed(0)

    params = deepseek_v3_mla
    cfg = params["cfg"]
    dtype = params["act_dtype"]
    N = cfg["num_attention_heads"]
    L = cfg["kv_lora_rank"]
    R = cfg["qk_rope_head_dim"]
    P = cfg["qk_nope_head_dim"]
    head_dim = L + R

    mesh = maybe_mesh(arch, N)

    assert seq_len % 32 == 0, "flash_mla_prefill requires seq_len % 32 == 0"
    tokens = users * seq_len
    blocks_per_user = (seq_len + BLOCK_SIZE - 1) // BLOCK_SIZE
    num_blocks = users * blocks_per_user

    # Random activations (the impl consumes no projection weights to make these);
    # random absorbed weights come from `params`.
    q_nope = torch.randn(tokens, N, P, dtype=dtype)
    q_pe = torch.randn(tokens, N, R, dtype=dtype)
    kv_c_normed = torch.randn(tokens, L, dtype=dtype)
    k_pe = torch.randn(tokens, 1, R, dtype=dtype)
    kv_cache = torch.zeros(num_blocks, 1, BLOCK_SIZE, head_dim, dtype=dtype)
    # cache_position: one entry per user (only its shape[0] is read for `users`).
    cache_position = torch.full((users,), seq_len - 1, dtype=torch.int32)
    # Each user gets `blocks_per_user` distinct, contiguous block ids.
    page_table = torch.arange(num_blocks, dtype=torch.int32).view(
        users, blocks_per_user
    )

    inputs = (q_nope, q_pe, kv_c_normed, k_pe, kv_cache, cache_position, page_table)

    golden, _ = run_mla_impl(torch.device("cpu"), params, inputs)

    device_out, _ = run_mla_impl(torch_xla.device(), params, inputs, mesh=mesh)
    torch_xla.sync()
    device_out = device_out.cpu()

    assert device_out.shape == golden.shape == (tokens, N * cfg["v_head_dim"])
    got = pcc(device_out, golden)
    assert got >= REQUIRED_PCC, f"MLA prefill PCC {got:.5f} < {REQUIRED_PCC}"


@pytest.mark.nightly
@parametrize_arch(["llmbox"])
@pytest.mark.parametrize("users, cur_pos", [(1, 31), (2, 48), (1, 96)])
def test_paged_mla_decode_impl(users, cur_pos, arch, deepseek_v3_mla):
    xr.set_device_type("TT")
    torch.manual_seed(0)

    params = deepseek_v3_mla
    cfg = params["cfg"]
    dtype = params["act_dtype"]
    N = cfg["num_attention_heads"]
    L = cfg["kv_lora_rank"]
    R = cfg["qk_rope_head_dim"]
    P = cfg["qk_nope_head_dim"]
    V = cfg["v_head_dim"]
    head_dim = L + R

    mesh = maybe_mesh(arch, N)

    # Enough blocks per user to hold the current position cur_pos.
    blocks_per_user = cur_pos // BLOCK_SIZE + 1
    num_blocks = users * blocks_per_user

    # One decode token per user; the prior context is the pre-seeded cache.
    q_nope = torch.randn(users, N, P, dtype=dtype)
    q_pe = torch.randn(users, N, R, dtype=dtype)
    kv_c_normed = torch.randn(users, L, dtype=dtype)
    k_pe = torch.randn(users, 1, R, dtype=dtype)
    seeded_cache = torch.randn(num_blocks, 1, BLOCK_SIZE, head_dim, dtype=dtype)
    cache_position = torch.full((users,), cur_pos, dtype=torch.int32)
    page_table = torch.arange(num_blocks, dtype=torch.int32).view(
        users, blocks_per_user
    )

    inputs = (q_nope, q_pe, kv_c_normed, k_pe, seeded_cache, cache_position, page_table)

    golden, cpu_cache = run_mla_impl(torch.device("cpu"), params, inputs)

    device_out, device_cache = run_mla_impl(
        torch_xla.device(), params, inputs, mesh=mesh
    )
    torch_xla.sync()
    device_out, device_cache = device_out.cpu(), device_cache.cpu()

    # -- Decode attention output (one token per user) --
    assert device_out.shape == golden.shape == (users, N * V)
    got = pcc(device_out, golden)
    assert got >= REQUIRED_PCC, f"MLA decode PCC {got:.5f} < {REQUIRED_PCC}"

    # -- Cache update: new token at cur_pos, prior context untouched --
    # `run_mla_impl` clones the cache, so `seeded_cache` still holds the pre-decode
    # state to compare the [0, cur_pos) prefix against.
    new_token_k = latent_k(kv_c_normed, k_pe)  # [users, head_dim]
    after = gather_cache(cpu_cache, page_table, cur_pos + 1)
    before = gather_cache(seeded_cache, page_table, cur_pos + 1)
    assert torch.allclose(
        after[:, cur_pos, :], new_token_k
    ), "decode token not written at cur_pos"
    assert torch.allclose(
        after[:, :cur_pos, :], before[:, :cur_pos, :]
    ), "decode clobbered prior context"
    cache_pcc = pcc(device_cache, cpu_cache)
    assert (
        cache_pcc >= REQUIRED_PCC
    ), f"decode cache PCC {cache_pcc:.5f} < {REQUIRED_PCC}"


@pytest.mark.nightly
@parametrize_arch(["llmbox"])
@pytest.mark.parametrize("users, seq_len", [(1, 64), (2, 64)])
def test_mla_prefill_and_decode_impl(users, seq_len, arch, deepseek_v3_mla):
    """
    Run MLA prefill (fills the paged latent cache) then a decode step against
    that cache, with random weights at the real DeepSeek-V3 dims + config,
    verifying the cache after each stage.
    """
    xr.set_device_type("TT")
    torch.manual_seed(0)

    params = deepseek_v3_mla
    cfg = params["cfg"]
    dtype = params["act_dtype"]
    N = cfg["num_attention_heads"]
    L = cfg["kv_lora_rank"]
    R = cfg["qk_rope_head_dim"]
    P = cfg["qk_nope_head_dim"]
    V = cfg["v_head_dim"]
    head_dim = L + R

    assert seq_len % 32 == 0, "flash_mla_prefill requires seq_len % 32 == 0"
    mesh = maybe_mesh(arch, N)

    # One extra block per user past the prefill length, to hold the decode token
    # written at position seq_len.
    blocks_per_user = seq_len // BLOCK_SIZE + 1
    num_blocks = users * blocks_per_user
    page_table = torch.arange(num_blocks, dtype=torch.int32).view(
        users, blocks_per_user
    )

    # ----- Prefill inputs (S = seq_len tokens per user) -----
    tokens = users * seq_len
    q_nope_p = torch.randn(tokens, N, P, dtype=dtype)
    q_pe_p = torch.randn(tokens, N, R, dtype=dtype)
    kv_c_p = torch.randn(tokens, L, dtype=dtype)
    k_pe_p = torch.randn(tokens, 1, R, dtype=dtype)
    kv_cache = torch.zeros(num_blocks, 1, BLOCK_SIZE, head_dim, dtype=dtype)
    cache_position_p = torch.full((users,), seq_len - 1, dtype=torch.int32)
    prefill_inputs = (
        q_nope_p,
        q_pe_p,
        kv_c_p,
        k_pe_p,
        kv_cache,
        cache_position_p,
        page_table,
    )

    # ----- Decode inputs (S = 1 token per user, written at position seq_len) ---
    q_nope_d = torch.randn(users, N, P, dtype=dtype)
    q_pe_d = torch.randn(users, N, R, dtype=dtype)
    kv_c_d = torch.randn(users, L, dtype=dtype)
    k_pe_d = torch.randn(users, 1, R, dtype=dtype)
    cache_position_d = torch.full((users,), seq_len, dtype=torch.int32)

    def _decode_inputs(cache_after_prefill):
        return (
            q_nope_d,
            q_pe_d,
            kv_c_d,
            k_pe_d,
            cache_after_prefill,
            cache_position_d,
            page_table,
        )

    # ----- CPU goldens: prefill, then decode against the prefilled cache -----
    golden_prefill, cpu_cache_prefill = run_mla_impl(
        torch.device("cpu"), params, prefill_inputs
    )
    golden_decode, _ = run_mla_impl(
        torch.device("cpu"), params, _decode_inputs(cpu_cache_prefill)
    )

    # ----- Device: prefill, then decode on device right after it -----
    device_prefill, device_cache_prefill = run_mla_impl(
        torch_xla.device(), params, prefill_inputs, mesh=mesh
    )
    torch_xla.sync()
    device_prefill = device_prefill.cpu()
    device_cache_prefill = device_cache_prefill.cpu()

    device_decode, device_cache_decode = run_mla_impl(
        torch_xla.device(), params, _decode_inputs(device_cache_prefill), mesh=mesh
    )
    torch_xla.sync()
    device_decode = device_decode.cpu()
    device_cache_decode = device_cache_decode.cpu()

    # ===== Prefill: attention output + cache filled with the prefill tokens ====
    assert device_prefill.shape == golden_prefill.shape == (tokens, N * V)
    got = pcc(device_prefill, golden_prefill)
    assert got >= REQUIRED_PCC, f"MLA prefill PCC {got:.5f} < {REQUIRED_PCC}"

    expected_prefill_k = latent_k(kv_c_p, k_pe_p).view(users, seq_len, head_dim)
    filled = gather_cache(device_cache_prefill, page_table, seq_len)
    assert torch.allclose(filled, expected_prefill_k), "prefill cache filled wrong"
    # The decode block (last per user) hasn't been written yet.
    assert (
        device_cache_prefill[page_table[:, -1].long()] == 0
    ).all(), "decode block not zero after prefill"

    # ===== Decode: attention output + cache appended with the new token ========
    assert device_decode.shape == golden_decode.shape == (users, N * V)
    got = pcc(device_decode, golden_decode)
    assert got >= REQUIRED_PCC, f"MLA decode PCC {got:.5f} < {REQUIRED_PCC}"

    expected_decode_k = latent_k(kv_c_d, k_pe_d)  # [users, head_dim]
    updated = gather_cache(device_cache_decode, page_table, seq_len + 1)
    assert torch.allclose(
        updated[:, :seq_len, :], expected_prefill_k
    ), "decode clobbered prefill context"
    assert torch.allclose(
        updated[:, seq_len, :], expected_decode_k
    ), "decode token not written at seq_len"
