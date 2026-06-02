# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""PCC test for ``TTMLAAttentionBackendImpl.forward`` (the unified MLA prefill
impl in ``vllm_tt.attention_mla``) using **real DeepSeek-V3 weights and config**
for a single layer.

What this exercises
-------------------
``TTMLAAttentionBackendImpl.forward`` is the heavy-lifting half of the MLA
prefill path: Q-absorption (``q_nope @ W_UK_T``), the latent flash attention
(``tt.flash_mla_prefill``), the ``W_UV`` projection back to physical space, and
the paged latent KV-cache write (``tt.paged_fill_cache``). The only *weights* it
consumes are ``layer.W_UK_T`` / ``layer.W_UV``, which vLLM derives from
``kv_b_proj.weight`` in ``MLAAttention.process_weights_after_loading``
(mla_attention.py:710-797). We load the real layer-0 ``kv_b_proj`` from
``deepseek-ai/DeepSeek-V3`` (FP8, block-dequantized) and derive the absorbed
weights exactly as vLLM does, and we take the MLA dims + attention scale from the
real V3 config. The activation inputs (q_nope/q_pe/kv_c_normed/k_pe) are random —
the function does not consume any projection weights to produce them.

Why test the impl directly (not the OOT wrapper)
------------------------------------------------
``forward`` only touches ``torch.ops.tt.*`` (dispatched by tensor device),
``layer.W_UK_T/W_UV`` (a stub here), and a ``TTMetadata`` we build directly. So
we can drive it without vLLM's platform/config/distributed/forward-context
machinery — keeping this a focused single-op-graph correctness check.

Sharding (tensor parallelism)
-----------------------------
On a multi-device arch (``llmbox``) the impl is run under SPMD with the N query
heads split across the mesh ``"model"`` axis — the standard head-parallel MLA
layout. The per-head Q activations (``q_nope``/``q_pe``) and the absorbed
weights (``W_UK_T``/``W_UV``, sharded on their head axis) partition along the
heads, so both einsums and the ``tt.flash_mla_prefill`` kernel run head-local
with no cross-device collectives. The compressed latent KV (``num_kv_heads ==
1``, i.e. ``kv_c_normed``/``k_pe`` and the paged latent cache) is shared by
every head and stays replicated — exactly how vLLM keeps the MLA KV cache under
TP. ``single_device`` runs the same code path unsharded (``mesh=None``).

PCC method: run the impl on CPU (custom-op CPU fallback = SDPA + manual paged
fill) as golden, then on the TT device (StableHLO custom calls → ttnn), and
compare the returned attention output with Pearson correlation. Sharding is a
layout hint, so the gathered device output must still match the replicated CPU
golden. The paged cache write is exercised but its result is discarded by
``forward`` (it returns the attention output), so only the attention math is
scored.
"""

import math

import pytest
import torch
import torch_xla
import torch_xla.distributed.spmd as xs
import torch_xla.runtime as xr
from infra.utilities.torch_multichip_utils import enable_spmd, get_mesh

from tests.utils import parametrize_arch

REPO = "deepseek-ai/DeepSeek-V3"
LAYER = 0
BLOCK_SIZE = 32  # TTMLAAttentionBackend.get_page_size
# bf16 flash attention + fp8-derived absorbed weights; the einsums run in fp32
# inside the impl, so the residual error is dominated by the bf16 kernel.
REQUIRED_PCC = 0.99


# --------------------------------------------------------------------------- #
# Real-weight loading helpers
# --------------------------------------------------------------------------- #
def _yarn_get_mscale(scale: float, mscale: float = 1.0) -> float:
    """Mirror ``vllm.model_executor.models.deepseek_v2.yarn_get_mscale``."""
    if scale <= 1:
        return 1.0
    return 0.1 * mscale * math.log(scale) + 1.0


def _dequant_fp8_blockwise(
    w_fp8: torch.Tensor, scale_inv: torch.Tensor, block: int = 128
) -> torch.Tensor:
    """DeepSeek FP8 block-wise dequant: ``w_bf16 = w_fp8 * weight_scale_inv``,
    with one scale per ``block x block`` tile (weight_block_size = [128, 128])."""
    w = w_fp8.to(torch.float32)
    rows, cols = w.shape
    s = scale_inv.to(torch.float32)
    s = s.repeat_interleave(block, dim=0).repeat_interleave(block, dim=1)
    return w * s[:rows, :cols]


def _mla_scale(cfg: dict) -> float:
    """Attention scale exactly as DeepseekV2MLAAttention computes it
    (deepseek_v2.py:862, 918-939): base ``qk_head_dim**-0.5`` times the squared
    YaRN mscale when rope scaling is yarn."""
    qk_head_dim = cfg["qk_nope_head_dim"] + cfg["qk_rope_head_dim"]
    scale = qk_head_dim**-0.5
    rope = cfg.get("rope_scaling") or {}
    if rope.get("type") == "yarn" or rope.get("rope_type") == "yarn":
        mscale_all_dim = float(rope.get("mscale_all_dim", 0.0))
        m = _yarn_get_mscale(rope["factor"], mscale_all_dim)
        scale = scale * m * m
    return float(scale)


def _load_absorbed_weights(cfg: dict, act_dtype: torch.dtype):
    """Fetch real layer-0 ``kv_b_proj`` from DeepSeek-V3, FP8-dequant it, and
    derive ``(W_UK_T, W_UV)`` exactly as
    ``MLAAttention.process_weights_after_loading`` (mla_attention.py:710-797)."""
    import json

    from huggingface_hub import hf_hub_download
    from safetensors import safe_open

    N = cfg["num_attention_heads"]
    L = cfg["kv_lora_rank"]
    P = cfg["qk_nope_head_dim"]
    V = cfg["v_head_dim"]

    index = json.load(open(hf_hub_download(REPO, "model.safetensors.index.json")))
    key = f"model.layers.{LAYER}.self_attn.kv_b_proj.weight"
    shard = index["weight_map"][key]
    shard_path = hf_hub_download(REPO, shard)  # cached after first download
    with safe_open(shard_path, framework="pt") as f:
        weight = f.get_tensor(key)  # fp8_e4m3, [N*(P+V), L]
        scale_inv = f.get_tensor(key + "_scale_inv")  # [ceil/128, ceil/128]

    # [N*(P+V), L] -> [L, N, P+V]  (mla_attention.py:714-732)
    kv_b = _dequant_fp8_blockwise(weight, scale_inv).to(act_dtype)
    kv_b = kv_b.t().contiguous().view(L, N, P + V)
    W_UK, W_UV = kv_b.split([P, V], dim=-1)  # [L,N,P], [L,N,V]
    # (mla_attention.py:794-797)
    W_UV = W_UV.transpose(0, 1).contiguous()  # [N, L, V]
    W_UK_T = W_UK.permute(1, 2, 0).contiguous()  # [N, P, L]
    return W_UK_T, W_UV


@pytest.fixture(scope="module")
def deepseek_v3_mla():
    """Load real DeepSeek-V3 config + absorbed layer-0 weights once. Skips the
    whole module if the checkpoint can't be reached (no network / no auth)."""
    act_dtype = torch.bfloat16
    try:
        import json

        from huggingface_hub import hf_hub_download

        cfg = json.load(open(hf_hub_download(REPO, "config.json")))
        W_UK_T, W_UV = _load_absorbed_weights(cfg, act_dtype)
    except Exception as e:  # network/auth/format issues -> skip, don't fail
        pytest.skip(f"DeepSeek-V3 weights/config unavailable: {type(e).__name__}: {e}")

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
    """Pearson correlation, mirroring TorchComparisonEvaluator.compute_pcc
    (tests/infra/evaluators/torch_comparison_evaluator.py:160-164)."""
    x = device_out.flatten().float()
    y = golden.flatten().float()
    if torch.allclose(x, y, rtol=1e-2, atol=1e-2):
        return 1.0
    vx, vy = x - x.mean(), y - y.mean()
    denom = vx.norm() * vy.norm()
    return 1.0 if denom == 0 else float((vx @ vy) / denom)


def _run_impl(device, params, inputs, mesh=None):
    """Build a fresh impl + stub layer + TTMetadata on ``device`` and run
    ``TTMLAAttentionBackendImpl.forward``; returns the attention output.

    When ``mesh`` is given (the multi-device path), the inputs and absorbed
    weights are head-parallel sharded across the mesh ``"model"`` axis before
    ``forward`` runs."""
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

    if mesh is not None:
        # Head-parallel (tensor-parallel) MLA sharding: split the N query heads
        # across the mesh "model" axis. The per-head Q activations and absorbed
        # weights shard along their head axis; the compressed latent KV
        # (num_kv_heads == 1) is shared by every head, so kv_c_normed / k_pe /
        # cache_position / page_table and the paged latent cache stay replicated.
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
    return impl.forward(
        q=(q_nope, q_pe),
        kv_c_normed=kv_c_normed,
        k_pe=k_pe,
        kv_cache=kv_cache,
        attn_metadata=attn_metadata,
        layer=layer,
        output=None,
    )


@pytest.mark.nightly
@parametrize_arch(["single_device", "llmbox"])
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

    if arch == "llmbox":
        # Head-parallel tensor parallelism: one shard of the N heads per device.
        enable_spmd()
        num_devices = xr.global_runtime_device_count()
        if N % num_devices != 0:
            pytest.skip(f"num_heads ({N}) not divisible by num_devices ({num_devices})")
        mesh = get_mesh((1, num_devices), ("batch", "model"))
    else:
        mesh = None

    assert seq_len % 32 == 0, "flash_mla_prefill requires seq_len % 32 == 0"
    tokens = users * seq_len
    blocks_per_user = (seq_len + BLOCK_SIZE - 1) // BLOCK_SIZE
    num_blocks = users * blocks_per_user

    # Random activations (the impl consumes no projection weights to make these);
    # real absorbed weights come from `params`.
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

    golden = _run_impl(torch.device("cpu"), params, inputs)

    device_out = _run_impl(torch_xla.device(), params, inputs, mesh=mesh)
    torch_xla.sync()
    device_out = device_out.cpu()

    assert device_out.shape == golden.shape == (tokens, N * cfg["v_head_dim"])
    pcc = _pcc(device_out, golden)
    assert pcc >= REQUIRED_PCC, f"MLA prefill PCC {pcc:.5f} < {REQUIRED_PCC}"
