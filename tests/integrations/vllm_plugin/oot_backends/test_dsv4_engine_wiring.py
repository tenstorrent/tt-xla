# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Full-engine wiring for DeepSeek-V4 on TT (device-light unit tests).

Covers the three plumbing pieces that let a DSV4 model reach the TT DSV4
attention backend + MoE without a real checkpoint or xla execution:

1. **Platform gating** — ``TTPlatform.get_attn_backend_cls`` routes a DSV4
   (sparse-MLA) model to ``TTDeepseekV4AttentionBackend`` while other sparse
   models still raise, and the non-sparse MLA path is unchanged.
2. **KV-cache spec** — ``TTModelRunner.get_kv_cache_spec`` emits a bf16
   ``MLAAttentionSpec`` for the separate ``DeepseekV4SWACache`` layer (block
   size 64, head_size = head_dim), *not* the upstream uint8 / 584-B
   ``SlidingWindowMLASpec``; the SWA-only ``DeepseekV4MLAAttention`` layer gets
   no cache of its own, and a compressed (C4A/C128A) layer raises.
3. **MoE routing** — ``TTFusedMoE`` reproduces DSV4's ``sqrtsoftplus`` + noaux_tc
   routing in torch (vLLM's own path is a CUDA-only kernel); plain-softmax
   models are unaffected.
"""
import contextlib
import math
from types import SimpleNamespace
from unittest import mock

import pytest
import torch
import torch.nn as nn

deepseek_v4_attention = pytest.importorskip(
    "vllm.model_executor.layers.deepseek_v4_attention"
)

_HEAD_DIM, _ROPE = 128, 64


# --------------------------------------------------------------------------- #
# Shared build helpers (duplicated per-file, matching the other DSV4 tests).
# --------------------------------------------------------------------------- #
def _tiny_dsv4_config():
    from vllm.transformers_utils.configs.deepseek_v4 import DeepseekV4Config

    return DeepseekV4Config(
        hidden_size=512,
        num_attention_heads=8,
        num_hidden_layers=2,
        head_dim=_HEAD_DIM,
        qk_rope_head_dim=_ROPE,
        kv_lora_rank=_HEAD_DIM,
        q_lora_rank=256,
        o_lora_rank=64,
        o_groups=1,
        sliding_window=64,
        compress_ratios=[1, 1],  # all SWA-only
        compress_rope_theta=10000.0,
        rope_theta=10000.0,
        rope_scaling=None,
        rope_parameters={"rope_type": "default"},
        rms_norm_eps=1e-6,
        max_position_embeddings=4096,
        index_topk=64,
        index_n_heads=8,
        index_head_dim=64,
        hc_mult=1,
        hc_sinkhorn_iters=1,
        hc_eps=1e-6,
        vocab_size=1000,
        quantization_config={"scale_fmt": None},
    )


def _mock_vllm_config(cfg):
    from vllm.config import CacheConfig, CompilationConfig

    cache_config = CacheConfig(block_size=64, cache_dtype="auto")
    comp_config = CompilationConfig()
    comp_config.custom_ops = ["none"]  # native CustomOp dispatch; no CUDA
    return SimpleNamespace(
        model_config=SimpleNamespace(
            hf_config=cfg, dtype=torch.bfloat16, max_model_len=4096
        ),
        cache_config=cache_config,
        scheduler_config=SimpleNamespace(max_num_batched_tokens=1024),
        compilation_config=comp_config,
        quant_config=None,
    )


@contextlib.contextmanager
def _construction_context(vllm_config):
    from vllm.config import set_current_vllm_config

    with contextlib.ExitStack() as stack:
        stack.enter_context(set_current_vllm_config(vllm_config))
        stack.enter_context(
            mock.patch.object(
                deepseek_v4_attention,
                "get_tensor_model_parallel_world_size",
                lambda: 1,
            )
        )
        yield


def _build_wrapper(vllm_config, prefix="model.layers.0.attn"):
    import vllm_tt.attention_impls.attention_dsv4  # noqa: F401  (registers OOT)
    from vllm.model_executor.layers.deepseek_v4_attention import (
        DeepseekV4MLAModules,
        DeepseekV4MultiHeadLatentAttentionWrapper,
    )

    cfg = vllm_config.model_config.hf_config
    padded_heads = 64
    mla_modules = DeepseekV4MLAModules(
        vllm_config=vllm_config,
        fused_wqa_wkv=nn.Identity(),
        q_norm=nn.Identity(),
        wq_b=nn.Identity(),
        kv_norm=nn.Identity(),
        wo_a=nn.Identity(),
        wo_b=nn.Identity(),
        attn_sink=torch.full((padded_heads,), float("-inf")),
        rotary_emb=SimpleNamespace(cos_sin_cache=None),
        indexer=None,
        indexer_rotary_emb=SimpleNamespace(cos_sin_cache=None),
        topk_indices_buffer=None,
        aux_stream_list=None,
    )
    return DeepseekV4MultiHeadLatentAttentionWrapper(
        hidden_size=cfg.hidden_size,
        num_heads=cfg.num_attention_heads,
        head_dim=_HEAD_DIM,
        scale=_HEAD_DIM**-0.5,
        qk_nope_head_dim=_HEAD_DIM - _ROPE,
        qk_rope_head_dim=_ROPE,
        v_head_dim=_HEAD_DIM,
        q_lora_rank=cfg.q_lora_rank,
        kv_lora_rank=_HEAD_DIM,
        o_lora_rank=cfg.o_lora_rank,
        mla_modules=mla_modules,
        window_size=cfg.sliding_window,
        compress_ratio=1,
        cache_config=vllm_config.cache_config,
        quant_config=None,
        prefix=prefix,
    )


# --------------------------------------------------------------------------- #
# 1. Platform gating
# --------------------------------------------------------------------------- #
def _selector(use_sparse=False, use_mla=False):
    from vllm.v1.attention.selector import AttentionSelectorConfig

    return AttentionSelectorConfig(
        head_size=_HEAD_DIM,
        dtype=torch.bfloat16,
        kv_cache_dtype=None,
        block_size=64,
        use_mla=use_mla,
        use_sparse=use_sparse,
    )


def _patch_architectures(monkeypatch, archs):
    monkeypatch.setattr(
        "vllm.config.get_current_vllm_config",
        lambda: SimpleNamespace(
            model_config=SimpleNamespace(hf_config=SimpleNamespace(architectures=archs))
        ),
    )


_DSV4_BACKEND_PATH = (
    "vllm_tt.attention_impls.attention_dsv4.TTDeepseekV4AttentionBackend"
)


@pytest.mark.push
def test_gating_routes_dsv4_sparse_to_dsv4_backend(monkeypatch):
    from vllm.v1.attention.backends.registry import AttentionBackendEnum
    from vllm_tt.platform import TTPlatform

    _patch_architectures(monkeypatch, ["DeepseekV4ForCausalLM"])
    path = TTPlatform.get_attn_backend_cls(
        AttentionBackendEnum.CUSTOM, _selector(use_sparse=True)
    )
    assert path == _DSV4_BACKEND_PATH


@pytest.mark.push
def test_gating_other_sparse_still_raises(monkeypatch):
    from vllm.v1.attention.backends.registry import AttentionBackendEnum
    from vllm_tt.platform import TTPlatform

    _patch_architectures(monkeypatch, ["DeepseekV32ForCausalLM"])
    with pytest.raises(NotImplementedError, match="Sparse Attention"):
        TTPlatform.get_attn_backend_cls(
            AttentionBackendEnum.CUSTOM, _selector(use_sparse=True)
        )


@pytest.mark.push
def test_gating_mla_path_unchanged():
    from vllm.v1.attention.backends.registry import AttentionBackendEnum
    from vllm_tt.platform import TTPlatform

    # use_mla (non-sparse) is unaffected by the DSV4 gate.
    path = TTPlatform.get_attn_backend_cls(
        AttentionBackendEnum.CUSTOM, _selector(use_mla=True)
    )
    assert path == AttentionBackendEnum.FLASH_ATTN_MLA.get_path()


# --------------------------------------------------------------------------- #
# 2. get_kv_cache_spec
# --------------------------------------------------------------------------- #
def _run_get_kv_cache_spec(monkeypatch, vllm_config, layers, dtype=torch.bfloat16):
    """Call the real ``TTModelRunner.get_kv_cache_spec`` with a fake ``self``
    and a stubbed layer map (avoids instantiating the heavy runner/device)."""
    import vllm_tt.model_runner as mr

    monkeypatch.setattr(mr, "get_layers_from_vllm_config", lambda cfg, base: layers)
    fake_self = SimpleNamespace(
        vllm_config=vllm_config,
        kv_cache_spec_dtype=dtype,
        shared_kv_cache_layers={},
    )
    return mr.TTModelRunner.get_kv_cache_spec(fake_self)


@pytest.mark.push
def test_kv_cache_spec_swa_only(monkeypatch):
    from vllm.v1.kv_cache_interface import MLAAttentionSpec

    cfg = _tiny_dsv4_config()
    vllm_config = _mock_vllm_config(cfg)
    with _construction_context(vllm_config):
        wrapper = _build_wrapper(vllm_config)

    swa = wrapper.swa_cache_layer
    layers = {swa.prefix: swa, "model.layers.0.attn.mla": wrapper.mla_attn}
    spec = _run_get_kv_cache_spec(monkeypatch, vllm_config, layers)

    # The SWA cache layer gets a bf16 MLA-latent spec (NOT the upstream uint8
    # SlidingWindowMLASpec), block_size 64, head_size == head_dim.
    assert swa.prefix in spec
    swa_spec = spec[swa.prefix]
    assert isinstance(swa_spec, MLAAttentionSpec)
    assert swa_spec.block_size == 64
    assert swa_spec.num_kv_heads == 1
    assert swa_spec.head_size == _HEAD_DIM
    assert swa_spec.dtype == torch.bfloat16

    # The SWA-only DeepseekV4MLAAttention layer holds no cache of its own.
    assert "model.layers.0.attn.mla" not in spec


@pytest.mark.push
def test_kv_cache_spec_compressed_layer_raises(monkeypatch):
    cfg = _tiny_dsv4_config()
    vllm_config = _mock_vllm_config(cfg)
    with _construction_context(vllm_config):
        wrapper = _build_wrapper(vllm_config)

    # A compressed branch (C4A/C128A) needs a second KV cache group — not
    # supported yet, so the spec builder must fail loudly rather than silently
    # allocating no cache.
    wrapper.mla_attn.compress_ratio = 128
    layers = {"model.layers.0.attn.mla": wrapper.mla_attn}
    with pytest.raises(NotImplementedError, match="compress_ratio"):
        _run_get_kv_cache_spec(monkeypatch, vllm_config, layers)


# --------------------------------------------------------------------------- #
# 3. MoE routing (TTFusedMoE)
# --------------------------------------------------------------------------- #
def _fake_moe(**kw):
    base = dict(
        top_k=2,
        renormalize=True,
        scoring_func="sqrtsoftplus",
        e_score_correction_bias=None,
        hash_indices_table=None,
        routed_scaling_factor=1.0,
    )
    base.update(kw)
    return SimpleNamespace(**base)


@pytest.mark.push
def test_moe_sqrtsoftplus_matches_reference_structure():
    from vllm_tt.layers.fused_moe import TTFusedMoE

    torch.manual_seed(0)
    logits = torch.randn(5, 8)
    fake = _fake_moe(top_k=3, renormalize=False, routed_scaling_factor=1.0)
    w, ids = TTFusedMoE._route_sqrtsoftplus(fake, logits)

    scores = torch.nn.functional.softplus(logits.float()).sqrt()
    exp_w, exp_ids = torch.topk(scores, 3, dim=-1)
    assert torch.equal(ids, exp_ids)
    assert torch.allclose(w, exp_w, atol=1e-6)


@pytest.mark.push
def test_moe_sqrtsoftplus_bias_selects_but_unbiased_weights():
    """noaux_tc: the bias steers *which* experts win, but the weight is the
    unbiased score (mirrors vLLM's grouped_topk / fused_topk_bias)."""
    from vllm_tt.layers.fused_moe import TTFusedMoE

    logits = torch.zeros(1, 4)
    bias = torch.tensor([10.0, 0.0, 0.0, 0.0])  # forces expert 0 into the top-k
    fake = _fake_moe(top_k=1, renormalize=False, e_score_correction_bias=bias)
    w, ids = TTFusedMoE._route_sqrtsoftplus(fake, logits)

    assert ids[0, 0].item() == 0
    # unbiased score of expert 0 = sqrt(softplus(0)) = sqrt(ln 2)
    assert abs(w[0, 0].item() - math.sqrt(math.log(2))) < 1e-4


@pytest.mark.push
def test_moe_sqrtsoftplus_renormalize_and_scaling():
    from vllm_tt.layers.fused_moe import TTFusedMoE

    torch.manual_seed(1)
    logits = torch.randn(3, 6)
    fake = _fake_moe(top_k=3, renormalize=True, routed_scaling_factor=2.5)
    w, _ = TTFusedMoE._route_sqrtsoftplus(fake, logits)
    # renormalize -> per-row sum 1, then * routed_scaling_factor.
    assert torch.allclose(w.sum(-1), torch.full((3,), 2.5), atol=1e-5)


@pytest.mark.push
def test_moe_hash_routing_selects_from_table():
    """DeepSeek-V4 hash layers: expert ids come from tid2eid[input_ids]; the
    gate logits are used only for the (unbiased sqrtsoftplus) weights."""
    from vllm_tt.layers.fused_moe import TTFusedMoE

    E, topk = 8, 2
    tid2eid = torch.tensor(
        [[0, 1], [2, 3], [4, 5], [6, 7], [1, 4]], dtype=torch.int32
    )  # [vocab=5, topk=2]
    fake = _fake_moe(
        top_k=topk,
        renormalize=True,
        hash_indices_table=tid2eid,
        routed_scaling_factor=1.0,
    )
    logits = torch.randn(3, E)
    input_ids = torch.tensor([0, 4, 2])
    w, ids = TTFusedMoE._route_sqrtsoftplus(fake, logits, input_ids)

    # Indices are exactly the hash-table rows for those tokens.
    assert torch.equal(ids, tid2eid[input_ids].long())
    # Weights are the renormalized unbiased sqrtsoftplus scores at those experts.
    scores = torch.nn.functional.softplus(logits.float()).sqrt()
    exp = scores.gather(-1, tid2eid[input_ids].long())
    exp = exp / exp.sum(-1, keepdim=True)
    assert torch.allclose(w, exp, atol=1e-6)


@pytest.mark.push
def test_moe_softmax_default_unchanged():
    """Non-DSV4 models keep the plain softmax + top-k router."""
    from vllm_tt.layers.fused_moe import TTFusedMoE

    torch.manual_seed(2)
    logits = torch.randn(4, 5)
    fake = _fake_moe(scoring_func="softmax", top_k=2, renormalize=True)
    w, ids = TTFusedMoE._route_native(fake, logits)

    scores = torch.softmax(logits.float(), dim=-1)
    exp_w, exp_ids = torch.topk(scores, 2, dim=-1)
    exp_w = exp_w / exp_w.sum(-1, keepdim=True).clamp(min=1e-9)
    assert torch.equal(ids, exp_ids)
    assert torch.allclose(w, exp_w, atol=1e-6)
