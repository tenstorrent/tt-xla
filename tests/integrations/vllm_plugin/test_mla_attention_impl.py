# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
import math

import pytest
import torch
import torch_xla
import torch_xla.distributed.spmd as xs
import torch_xla.runtime as xr
from infra.utilities.torch_multichip_utils import enable_spmd, get_mesh

from tests.utils import parametrize_arch

BLOCK_SIZE = 32  # TTMLAAttentionBackend.get_page_size
REQUIRED_PCC = 0.99

_DEEPSEEK_V3_CFG = {
    "num_attention_heads": 128,
    "kv_lora_rank": 512,
    "qk_nope_head_dim": 128,
    "qk_rope_head_dim": 64,
    "v_head_dim": 128,
    "q_lora_rank": 1536,
    "hidden_size": 7168,
    "rope_scaling": {
        "beta_fast": 32,
        "beta_slow": 1,
        "factor": 40,
        "mscale": 1.0,
        "mscale_all_dim": 1.0,
        "original_max_position_embeddings": 4096,
        "type": "yarn",
    },
}


def _yarn_get_mscale(scale: float, mscale: float = 1.0) -> float:
    """
    Mirror ``vllm.model_executor.models.deepseek_v2.yarn_get_mscale``.
    """
    if scale <= 1:
        return 1.0
    return 0.1 * mscale * math.log(scale) + 1.0


def _mla_scale(cfg: dict) -> float:
    """
    Attention scale exactly as DeepseekV2MLAAttention computes it
    """
    qk_head_dim = cfg["qk_nope_head_dim"] + cfg["qk_rope_head_dim"]
    scale = qk_head_dim**-0.5
    rope = cfg.get("rope_scaling") or {}
    if rope.get("type") == "yarn" or rope.get("rope_type") == "yarn":
        mscale_all_dim = float(rope.get("mscale_all_dim", 0.0))
        m = _yarn_get_mscale(rope["factor"], mscale_all_dim)
        scale = scale * m * m
    return float(scale)


def _random_absorbed_weights(cfg: dict, act_dtype: torch.dtype):
    N = cfg["num_attention_heads"]
    L = cfg["kv_lora_rank"]
    P = cfg["qk_nope_head_dim"]
    V = cfg["v_head_dim"]

    weight = torch.randn(N * (P + V), L, dtype=torch.float32) / math.sqrt(L)

    # [N*(P+V), L] -> [L, N, P+V]
    kv_b = weight.to(act_dtype)
    kv_b = kv_b.t().contiguous().view(L, N, P + V)
    W_UK, W_UV = kv_b.split([P, V], dim=-1)  # [L,N,P], [L,N,V]
    W_UV = W_UV.transpose(0, 1).contiguous()  # [N, L, V]
    W_UK_T = W_UK.permute(1, 2, 0).contiguous()  # [N, P, L]
    return W_UK_T, W_UV


@pytest.fixture(scope="module")
def deepseek_v3_mla():
    """
    Fake DeepSeek-V3 MLA params: the real config dims/scale, but randomly
    generated absorbed weights
    """
    act_dtype = torch.bfloat16
    cfg = dict(_DEEPSEEK_V3_CFG)
    torch.manual_seed(0)
    W_UK_T, W_UV = _random_absorbed_weights(cfg, act_dtype)

    return {
        "cfg": cfg,
        "act_dtype": act_dtype,
        "scale": _mla_scale(cfg),
        "W_UK_T": W_UK_T,
        "W_UV": W_UV,
    }


# --------------------------------------------------------------------------- #
# Test
# --------------------------------------------------------------------------- #
def _pcc(device_out: torch.Tensor, golden: torch.Tensor) -> float:
    x = device_out.flatten().float()
    y = golden.flatten().float()
    if torch.allclose(x, y, rtol=1e-2, atol=1e-2):
        return 1.0
    vx, vy = x - x.mean(), y - y.mean()
    denom = vx.norm() * vy.norm()
    return 1.0 if denom == 0 else float((vx @ vy) / denom)


def _run_impl(device, params, inputs, mesh=None):
    """
    Build a fresh impl + stub layer + TTMetadata on ``device`` and run
    ``TTMLAAttentionBackendImpl.forward``; returns ``(attention_output,
    kv_cache)``
    """
    from types import SimpleNamespace

    from vllm_tt.attention import TTMetadata
    from vllm_tt.attention_mla import TTMLAAttentionBackendImpl

    cfg = params["cfg"]
    L = cfg["kv_lora_rank"]
    R = cfg["qk_rope_head_dim"]

    impl = TTMLAAttentionBackendImpl(
        num_heads=cfg["num_attention_heads"],
        head_size=L + R,  # MLA latent kv-cache head dim
        scale=params["scale"],
        num_kv_heads=1,
        alibi_slopes=None,
        sliding_window=None,
        kv_cache_dtype="auto",
        logits_soft_cap=None,
        attn_type="decoder",
        kv_sharing_target_layer_name=None,
        q_lora_rank=cfg.get("q_lora_rank"),
        kv_lora_rank=L,
        qk_nope_head_dim=cfg["qk_nope_head_dim"],
        qk_rope_head_dim=R,
        qk_head_dim=cfg["qk_nope_head_dim"] + R,
        v_head_dim=cfg["v_head_dim"],
    )

    # Stub `layer`: forward only reads layer.W_UK_T / layer.W_UV.
    W_UK_T = params["W_UK_T"].to(device)  # [N, P, L]
    W_UV = params["W_UV"].to(device)  # [N, L, V]

    q_nope, q_pe, kv_c_normed, k_pe, kv_cache, cache_position, page_table = (
        t.to(device) for t in inputs
    )
    # Mutated in place by forward; clone so the passed-in cache is preserved.
    kv_cache = kv_cache.clone()

    if mesh is not None:
        xs.mark_sharding(q_nope, mesh, (None, "model", None))  # [tokens, N, P]
        xs.mark_sharding(q_pe, mesh, (None, "model", None))  # [tokens, N, R]
        xs.mark_sharding(W_UK_T, mesh, ("model", None, None))  # [N, P, L]
        xs.mark_sharding(W_UV, mesh, ("model", None, None))  # [N, L, V]

    layer = SimpleNamespace(W_UK_T=W_UK_T, W_UV=W_UV)

    attn_metadata = TTMetadata(
        cache_position=cache_position,
        attn_mask=None,
        page_table=page_table,
        is_causal=True,
        fill_page_table=page_table,
    )
    out = impl.forward(
        q=(q_nope, q_pe),
        kv_c_normed=kv_c_normed,
        k_pe=k_pe,
        kv_cache=kv_cache,
        attn_metadata=attn_metadata,
        layer=layer,
        output=None,
    )
    return out, kv_cache


def _maybe_mesh(arch, num_heads):
    if arch != "llmbox":
        return None
    enable_spmd()
    num_devices = xr.global_runtime_device_count()
    if num_heads % num_devices != 0:
        pytest.skip(
            f"num_heads ({num_heads}) not divisible by num_devices ({num_devices})"
        )
    return get_mesh((1, num_devices), ("batch", "model"))


def _latent_k(kv_c_normed: torch.Tensor, k_pe: torch.Tensor) -> torch.Tensor:
    """
    The latent K the impl persists into the paged cache:
    (kv_c_normed: [tokens, L], k_pe: [tokens, 1, R]) -> [tokens, L + R].
    """
    return torch.cat([kv_c_normed, k_pe.squeeze(1)], dim=-1)


def _gather_cache(
    cache: torch.Tensor, page_table: torch.Tensor, num_positions: int
) -> torch.Tensor:
    """Read logical sequence positions ``[0, num_positions)`` out of the paged
    latent cache for every user, undoing the paging via ``page_table``.

    cache: [num_blocks, 1, BLOCK_SIZE, head_dim] -> [users, num_positions, head_dim].
    Position ``p`` of user ``u`` lives in physical block
    ``page_table[u, p // BLOCK_SIZE]`` at offset ``p % BLOCK_SIZE``."""
    users = page_table.shape[0]
    out = torch.empty(users, num_positions, cache.shape[-1], dtype=cache.dtype)
    for u in range(users):
        for p in range(num_positions):
            blk = int(page_table[u, p // BLOCK_SIZE])
            out[u, p] = cache[blk, 0, p % BLOCK_SIZE, :]
    return out


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

    mesh = _maybe_mesh(arch, N)

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

    golden, _ = _run_impl(torch.device("cpu"), params, inputs)

    device_out, _ = _run_impl(torch_xla.device(), params, inputs, mesh=mesh)
    torch_xla.sync()
    device_out = device_out.cpu()

    assert device_out.shape == golden.shape == (tokens, N * cfg["v_head_dim"])
    pcc = _pcc(device_out, golden)
    assert pcc >= REQUIRED_PCC, f"MLA prefill PCC {pcc:.5f} < {REQUIRED_PCC}"


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

    mesh = _maybe_mesh(arch, N)

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

    golden, cpu_cache = _run_impl(torch.device("cpu"), params, inputs)

    device_out, device_cache = _run_impl(torch_xla.device(), params, inputs, mesh=mesh)
    torch_xla.sync()
    device_out, device_cache = device_out.cpu(), device_cache.cpu()

    # -- Decode attention output (one token per user) --
    assert device_out.shape == golden.shape == (users, N * V)
    pcc = _pcc(device_out, golden)
    assert pcc >= REQUIRED_PCC, f"MLA decode PCC {pcc:.5f} < {REQUIRED_PCC}"

    # -- Cache update: new token at cur_pos, prior context untouched --
    # `_run_impl` clones the cache, so `seeded_cache` still holds the pre-decode
    # state to compare the [0, cur_pos) prefix against.
    new_token_k = _latent_k(kv_c_normed, k_pe)  # [users, head_dim]
    after = _gather_cache(cpu_cache, page_table, cur_pos + 1)
    before = _gather_cache(seeded_cache, page_table, cur_pos + 1)
    assert torch.allclose(
        after[:, cur_pos, :], new_token_k
    ), "decode token not written at cur_pos"
    assert torch.allclose(
        after[:, :cur_pos, :], before[:, :cur_pos, :]
    ), "decode clobbered prior context"
    cache_pcc = _pcc(device_cache, cpu_cache)
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
    mesh = _maybe_mesh(arch, N)

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
    golden_prefill, cpu_cache_prefill = _run_impl(
        torch.device("cpu"), params, prefill_inputs
    )
    golden_decode, _ = _run_impl(
        torch.device("cpu"), params, _decode_inputs(cpu_cache_prefill)
    )

    # ----- Device: prefill, then decode on device right after it -----
    device_prefill, device_cache_prefill = _run_impl(
        torch_xla.device(), params, prefill_inputs, mesh=mesh
    )
    torch_xla.sync()
    device_prefill = device_prefill.cpu()
    device_cache_prefill = device_cache_prefill.cpu()

    device_decode, device_cache_decode = _run_impl(
        torch_xla.device(), params, _decode_inputs(device_cache_prefill), mesh=mesh
    )
    torch_xla.sync()
    device_decode = device_decode.cpu()
    device_cache_decode = device_cache_decode.cpu()

    # ===== Prefill: attention output + cache filled with the prefill tokens ====
    assert device_prefill.shape == golden_prefill.shape == (tokens, N * V)
    pcc = _pcc(device_prefill, golden_prefill)
    assert pcc >= REQUIRED_PCC, f"MLA prefill PCC {pcc:.5f} < {REQUIRED_PCC}"

    expected_prefill_k = _latent_k(kv_c_p, k_pe_p).view(users, seq_len, head_dim)
    filled = _gather_cache(device_cache_prefill, page_table, seq_len)
    assert torch.allclose(filled, expected_prefill_k), "prefill cache filled wrong"
    # The decode block (last per user) hasn't been written yet.
    assert (
        device_cache_prefill[page_table[:, -1].long()] == 0
    ).all(), "decode block not zero after prefill"

    # ===== Decode: attention output + cache appended with the new token ========
    assert device_decode.shape == golden_decode.shape == (users, N * V)
    pcc = _pcc(device_decode, golden_decode)
    assert pcc >= REQUIRED_PCC, f"MLA decode PCC {pcc:.5f} < {REQUIRED_PCC}"

    expected_decode_k = _latent_k(kv_c_d, k_pe_d)  # [users, head_dim]
    updated = _gather_cache(device_cache_decode, page_table, seq_len + 1)
    assert torch.allclose(
        updated[:, :seq_len, :], expected_prefill_k
    ), "decode clobbered prefill context"
    assert torch.allclose(
        updated[:, seq_len, :], expected_decode_k
    ), "decode token not written at seq_len"
