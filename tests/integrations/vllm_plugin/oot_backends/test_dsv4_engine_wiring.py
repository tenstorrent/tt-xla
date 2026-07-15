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
def test_kv_cache_spec_c128a(monkeypatch):
    """A C128A (compress_ratio=128) DeepseekV4MLAAttention layer emits a second,
    distinct bf16 MLA cache group (the compressed latent), keyed by its own
    prefix, alongside the SWA group. The two must NOT merge (different
    compress_ratio) so the runner allocates them as separate cache groups."""
    from vllm.v1.kv_cache_interface import MLAAttentionSpec

    cfg = _tiny_dsv4_config()
    vllm_config = _mock_vllm_config(cfg)
    with _construction_context(vllm_config):
        wrapper = _build_wrapper(vllm_config)

    wrapper.mla_attn.compress_ratio = 128
    swa = wrapper.swa_cache_layer
    layers = {swa.prefix: swa, "model.layers.0.attn.mla": wrapper.mla_attn}
    spec = _run_get_kv_cache_spec(monkeypatch, vllm_config, layers)

    # SWA group (compress_ratio 1) + compressed group (compress_ratio 128).
    assert swa.prefix in spec and "model.layers.0.attn.mla" in spec
    comp_spec = spec["model.layers.0.attn.mla"]
    assert isinstance(comp_spec, MLAAttentionSpec)
    assert comp_spec.compress_ratio == 128
    assert comp_spec.num_kv_heads == 1
    assert comp_spec.head_size == _HEAD_DIM
    assert comp_spec.dtype == torch.bfloat16
    # block_size (token units) = 64*ratio so storage_block_size == 64 (>0).
    assert comp_spec.block_size == 64 * 128
    assert comp_spec.storage_block_size == 64
    # Distinct group: the SWA spec (compress_ratio 1) and the compressed spec
    # (compress_ratio 128) must not be mergeable.
    swa_spec = spec[swa.prefix]
    assert swa_spec.compress_ratio != comp_spec.compress_ratio


@pytest.mark.push
def test_kv_cache_spec_c4a_raises(monkeypatch):
    """C4A (compress_ratio=4) also needs the lightning-indexer branch + its own
    cache — not implemented on TT yet, so the spec builder fails loudly."""
    cfg = _tiny_dsv4_config()
    vllm_config = _mock_vllm_config(cfg)
    with _construction_context(vllm_config):
        wrapper = _build_wrapper(vllm_config)

    wrapper.mla_attn.compress_ratio = 4
    layers = {"model.layers.0.attn.mla": wrapper.mla_attn}
    with pytest.raises(NotImplementedError, match="C4A"):
        _run_get_kv_cache_spec(monkeypatch, vllm_config, layers)


# --------------------------------------------------------------------------- #
# 2b. Multi-group plumbing: hybrid-KV scoping, per-group metadata, init
# --------------------------------------------------------------------------- #
@pytest.mark.push
def test_support_hybrid_kv_cache_scoping():
    """The hybrid KV-cache manager is enabled ONLY for a DSV4 model that actually
    has a compressed layer (some compress_ratio > 1) — those need >1 KV-cache
    group. Pure-SWA DSV4 (all ratios <= 1) and non-DSV4 stay single-group /
    non-hybrid, so their scheduling is byte-for-byte unchanged."""
    from vllm_tt.platform import TTPlatform

    def hf(archs, ratios):
        return SimpleNamespace(architectures=archs, compress_ratios=ratios)

    assert TTPlatform._dsv4_needs_hybrid_kv(hf(["DeepseekV4ForCausalLM"], [0, 0, 128]))
    assert TTPlatform._dsv4_needs_hybrid_kv(hf(["DeepseekV4ForCausalLM"], [0, 4]))
    assert not TTPlatform._dsv4_needs_hybrid_kv(hf(["DeepseekV4ForCausalLM"], [0, 0]))
    assert not TTPlatform._dsv4_needs_hybrid_kv(hf(["DeepseekV4ForCausalLM"], [1, 1]))
    assert not TTPlatform._dsv4_needs_hybrid_kv(hf(["LlamaForCausalLM"], []))
    assert not TTPlatform._dsv4_needs_hybrid_kv(None)

    # the no-arg classmethod (called by vLLM during VllmConfig __post_init__)
    # just reflects the flag check_and_update_config stashed.
    saved = TTPlatform._dsv4_hybrid_kv
    try:
        TTPlatform._dsv4_hybrid_kv = True
        assert TTPlatform.support_hybrid_kv_cache() is True
        TTPlatform._dsv4_hybrid_kv = False
        assert TTPlatform.support_hybrid_kv_cache() is False
    finally:
        TTPlatform._dsv4_hybrid_kv = saved


@pytest.mark.push
def test_metadata_per_group():
    """The DSV4 compressed group's TTMetadata uses its OWN scheduler block table
    (input_batch.block_table[gi], NOT the window group's) and a compressed
    cache_position = (token_pos + 1)//ratio - 1 (one latent per `ratio` tokens)."""
    import vllm_tt.model_runner as mr
    from vllm_tt.attention_impls.attention import TTMetadata

    ratio, gi = 128, 1
    comp_bt = torch.tensor([[7], [9]], dtype=torch.int32)  # 2 reqs, 1 compressed blk

    class _BT:
        def get_cpu_tensor(self):
            return comp_bt

    fake_self = SimpleNamespace(
        _dsv4_comp_group=(gi, ratio, 64 * ratio),
        # block_table[0] is the window group (must NOT be used here); [gi] is comp
        input_batch=SimpleNamespace(block_table=[object(), _BT()]),
        block_table_cpu=torch.zeros(4, 3, dtype=torch.int32),
        device="cpu",
        dp_size=1,
    )
    swa_md = TTMetadata(
        cache_position=torch.tensor([255, 130], dtype=torch.int32),
        batch_idx=torch.tensor([0, 1], dtype=torch.int32),
        page_table=torch.zeros(2, 3, dtype=torch.int32),
    )
    md = mr.TTModelRunner._build_compressed_metadata(
        fake_self, swa_md, actual_num_reqs=2, target_num_reqs=2
    )
    # compressed cache_position: (255+1)//128-1 == 1 ; (130+1)//128-1 == 0
    assert md.cache_position.cpu().tolist() == [1, 0]
    # page_table is the COMPRESSED group's block table, not the window group's
    assert md.page_table[:2].cpu().tolist() == comp_bt.tolist()
    assert md.batch_idx is swa_md.batch_idx


@pytest.mark.push
def test_initialize_kv_cache_multi_group(monkeypatch):
    """initialize_kv_cache on a 2-group hybrid KVCacheConfig (SWA + compressed
    sharing ONE pooled tensor): detect the compressed group, split the shared
    pool per-layer (size // #layers — NOT the whole pool, which would OOM), keep
    the window group at self.block_size while the compressed group takes its own,
    and allocate one MLA-latent tensor per layer."""
    import vllm_tt.model_runner as mr
    from vllm.v1.kv_cache_interface import (
        KVCacheConfig,
        KVCacheGroupSpec,
        KVCacheTensor,
        MLAAttentionSpec,
    )

    HEAD, WIN_BS, RATIO = _HEAD_DIM, 32, 128
    swa_spec = MLAAttentionSpec(
        block_size=64,
        num_kv_heads=1,
        head_size=HEAD,
        dtype=torch.bfloat16,
        compress_ratio=1,
    )
    comp_spec = MLAAttentionSpec(
        block_size=64 * RATIO,
        num_kv_heads=1,
        head_size=HEAD,
        dtype=torch.bfloat16,
        compress_ratio=RATIO,
    )
    groups = [
        KVCacheGroupSpec(layer_names=["m.l0.swa"], kv_cache_spec=swa_spec),
        KVCacheGroupSpec(layer_names=["m.l0.mla"], kv_cache_spec=comp_spec),
    ]
    page = swa_spec.page_size_bytes  # 16384; comp storage_block == 64 -> same page
    n_blocks = 8
    pool = page * n_blocks * 2  # ONE tensor shared by the 2 layers
    config = KVCacheConfig(
        num_blocks=n_blocks,
        kv_cache_tensors=[KVCacheTensor(size=pool, shared_by=["m.l0.swa", "m.l0.mla"])],
        kv_cache_groups=groups,
    )

    captured = {}
    monkeypatch.setattr(
        mr, "bind_kv_cache", lambda kvc, ctx, run: captured.update(kvc=kvc)
    )
    monkeypatch.setattr(mr, "_dsv4_layer_classes", lambda: (object, object))
    monkeypatch.setattr(mr, "_get_layer_kv_cache_spec", lambda gspec, ln: gspec)
    monkeypatch.setattr(mr, "has_kv_transfer_group", lambda: False)

    def _fake_input_batch(**kw):
        bs = kw["block_sizes"]
        return SimpleNamespace(
            block_table=[
                SimpleNamespace(
                    get_cpu_tensor=lambda: torch.zeros(1, 1, dtype=torch.int32)
                )
                for _ in bs
            ]
        )

    monkeypatch.setattr(mr, "InputBatch", _fake_input_batch)

    fake_self = SimpleNamespace(
        block_size=WIN_BS,
        max_num_reqs=4,
        max_model_len=160,
        max_num_tokens=160,
        pin_memory=False,
        model_config=SimpleNamespace(get_vocab_size=lambda: 1000),
        input_batch=SimpleNamespace(
            block_table=[
                SimpleNamespace(
                    get_cpu_tensor=lambda: torch.zeros(1, 1, dtype=torch.int32)
                )
            ]
        ),
        block_table_cpu=torch.zeros(4, 5, dtype=torch.int32),
        device="cpu",
        kv_cache_dtype=torch.bfloat16,
        enable_tensor_parallel=False,
        parallel_mode=None,
        vllm_config=SimpleNamespace(
            compilation_config=SimpleNamespace(static_forward_context={})
        ),
        kv_caches=[],
        maybe_setup_cross_layer_kv_sharing=lambda kvc, cfg: None,
    )

    mr.TTModelRunner.initialize_kv_cache(fake_self, config)

    # multi-group detected; compressed group is index 1 (ratio 128).
    assert fake_self._is_dsv4 is True
    assert fake_self._is_multi_group is True
    assert fake_self._dsv4_comp_group == (1, RATIO, 64 * RATIO)
    # window group keeps self.block_size; compressed group takes its own.
    assert fake_self.group_block_sizes == [WIN_BS, 64 * RATIO]
    # each layer got its OWN latent tensor sized from pool//2 (not the whole pool).
    kvc = captured["kvc"]
    assert set(kvc) == {"m.l0.swa", "m.l0.mla"}
    per_layer_blocks = (pool // 2) // page
    for name, t in kvc.items():
        assert t.shape[0] == per_layer_blocks, (name, t.shape)
        assert t.ndim == 4 and t.dtype == torch.bfloat16


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
