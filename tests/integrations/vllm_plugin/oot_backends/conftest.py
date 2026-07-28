# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Shared helpers for the OOT MLA / DSA attention-impl tests.

``oot_backends/`` has no ``__init__.py``, so pytest puts this directory on
``sys.path`` and the test modules import these by bare name
(``from conftest import run_mla_impl, ...``) — the same pattern
``generative/conftest.py`` uses for ``assert_output_coherent`` and friends.

The impl under test is driven directly (no vLLM engine): every test builds a
``TTMLAAttentionBackendImpl`` plus a ``SimpleNamespace`` stub layer and a
``TTMetadata``, runs it on ``torch.device("cpu")`` for the golden and on
``torch_xla.device()`` for the device result, and compares by PCC.
"""

import math
from types import SimpleNamespace

import pytest
import torch
import torch_xla.distributed.spmd as xs
import torch_xla.runtime as xr
from infra.utilities.torch_multichip_utils import enable_spmd, get_mesh

BLOCK_SIZE = 32  # TTMLAAttentionBackend.get_page_size
REQUIRED_PCC = 0.99

DEEPSEEK_V3_CFG = {
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

# DeepSeek-V3.2 adds the sparse-attention indexer on top of the V3 MLA dims.
DEEPSEEK_V32_CFG = {
    **DEEPSEEK_V3_CFG,
    "index_head_dim": 128,
    "index_n_heads": 64,
    "index_topk": 2048,
}


def _yarn_get_mscale(scale: float, mscale: float = 1.0) -> float:
    """
    Mirror ``vllm.model_executor.models.deepseek_v2.yarn_get_mscale``.
    """
    if scale <= 1:
        return 1.0
    return 0.1 * mscale * math.log(scale) + 1.0


def mla_scale(cfg: dict) -> float:
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


def random_absorbed_weights(cfg: dict, act_dtype: torch.dtype):
    """Random ``(W_UK_T, W_UV)`` in the layout ``MLAAttention`` produces.

    Mirrors ``process_weights_after_loading``: kv_b_proj's ``[N*(P+V), L]``
    weight is split into W_UK / W_UV and permuted to ``[N, P, L]`` / ``[N, L, V]``.
    """
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


def _mla_params(cfg: dict) -> dict:
    act_dtype = torch.bfloat16
    torch.manual_seed(0)
    W_UK_T, W_UV = random_absorbed_weights(cfg, act_dtype)
    return {
        "cfg": dict(cfg),
        "act_dtype": act_dtype,
        "scale": mla_scale(cfg),
        "W_UK_T": W_UK_T,
        "W_UV": W_UV,
    }


@pytest.fixture(scope="module")
def deepseek_v3_mla():
    """
    Fake DeepSeek-V3 MLA params: the real config dims/scale, but randomly
    generated absorbed weights
    """
    return _mla_params(DEEPSEEK_V3_CFG)


@pytest.fixture(scope="module")
def deepseek_v32_mla():
    """DeepSeek-V3.2 MLA params (V3 dims plus the indexer config)."""
    return _mla_params(DEEPSEEK_V32_CFG)


def pcc(device_out: torch.Tensor, golden: torch.Tensor) -> float:
    x = device_out.flatten().float()
    y = golden.flatten().float()
    if torch.allclose(x, y, rtol=1e-2, atol=1e-2):
        return 1.0
    vx, vy = x - x.mean(), y - y.mean()
    denom = vx.norm() * vy.norm()
    return 1.0 if denom == 0 else float((vx @ vy) / denom)


def run_mla_impl(device, params, inputs, mesh=None, indexer=None):
    """Build a fresh impl + stub layer + TTMetadata on ``device`` and run
    ``TTMLAAttentionBackendImpl.forward``; returns ``(attention_output, kv_cache)``.

    ``forward`` writes into a caller-allocated ``output`` and returns ``None``, so
    the output tensor is allocated here and returned instead.

    ``indexer``, when given, is a stub exposing ``topk_indices`` / ``topk_tokens``
    / ``k_chunk_size``; it is attached both to the impl (static config) and to the
    stub layer (per-step indices), matching how vLLM wires the real ``Indexer``.
    """
    from vllm_tt.attention_impls.attention import TTMetadata
    from vllm_tt.attention_impls.attention_mla import TTMLAAttentionBackendImpl

    cfg = params["cfg"]
    N = cfg["num_attention_heads"]
    L = cfg["kv_lora_rank"]
    R = cfg["qk_rope_head_dim"]
    V = cfg["v_head_dim"]

    impl = TTMLAAttentionBackendImpl(
        num_heads=N,
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
        v_head_dim=V,
        indexer=indexer,
    )

    # Stub `layer`: forward only reads layer.W_UK_T / layer.W_UV / layer.indexer.
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

    layer = SimpleNamespace(W_UK_T=W_UK_T, W_UV=W_UV, indexer=indexer)

    attn_metadata = TTMetadata(
        cache_position=cache_position,
        attn_mask=None,
        page_table=page_table,
        is_causal=True,
        fill_page_table=page_table,
    )

    output = torch.empty(
        (q_nope.shape[0], N * V), dtype=params["act_dtype"], device=device
    )
    impl.forward(
        q=(q_nope, q_pe),
        kv_c_normed=kv_c_normed,
        k_pe=k_pe,
        kv_cache=kv_cache,
        attn_metadata=attn_metadata,
        layer=layer,
        output=output,
    )
    return output, kv_cache


def maybe_mesh(arch, num_heads):
    if arch != "llmbox":
        return None
    enable_spmd()
    num_devices = xr.global_runtime_device_count()
    if num_heads % num_devices != 0:
        pytest.skip(
            f"num_heads ({num_heads}) not divisible by num_devices ({num_devices})"
        )
    return get_mesh((1, num_devices), ("batch", "model"))


def latent_k(kv_c_normed: torch.Tensor, k_pe: torch.Tensor) -> torch.Tensor:
    """
    The latent K the impl persists into the paged cache:
    (kv_c_normed: [tokens, L], k_pe: [tokens, 1, R]) -> [tokens, L + R].
    """
    return torch.cat([kv_c_normed, k_pe.squeeze(1)], dim=-1)


def gather_cache(
    cache: torch.Tensor, page_table: torch.Tensor, num_positions: int
) -> torch.Tensor:
    """Read logical sequence positions ``[0, num_positions)`` out of the paged
    cache for every user, undoing the paging via ``page_table``.

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
