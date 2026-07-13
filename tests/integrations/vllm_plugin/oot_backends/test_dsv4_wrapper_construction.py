# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Construction test for the DeepSeek-V4 OOT layer wrapper (``TTDeepseekV4MLAWrapper``).

vLLM's ``DeepseekV4MultiHeadLatentAttentionWrapper`` is a ``PluggableLayer`` whose
``__init__`` is hard-bound to CUDA (asserts ``get_device_capability() is not
None``, allocates ``torch.cuda.Event``, and its ``DeepseekV4MLAAttention`` sub-layer
asserts an fp8 kv-cache). This test proves the TT OOT replacement — registered via
``register_oot`` and running the base ctor under temporary monkeypatches
(the same technique tpu-inference uses) — **constructs on the tt platform**.

It is a device-light unit test: construction is module init (the CUDA-only calls
are the blockers, all neutralized by the recipe); no xla execution is needed.
``forward`` is still a documented ``NotImplementedError`` — the bf16
preprocess/attention/o-proj reimplementation is the next, separately-validated
step — so this test also pins that current state.
"""
import contextlib
from types import SimpleNamespace
from unittest import mock

import pytest
import torch
import torch.nn as nn

deepseek_v4_attention = pytest.importorskip(
    "vllm.model_executor.layers.deepseek_v4_attention"
)

_HEAD_DIM, _ROPE = 128, 64


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
    """Set the current vLLM config + stub the TP-size call (needs a distributed
    group we don't init) so the wrapper's construction path runs standalone."""
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


def _build_wrapper(vllm_config):
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
    # Construct by the *base* class name; PluggableLayer.__new__ dispatches to the
    # registered OOT subclass (TTDeepseekV4MLAWrapper) and runs its __init__.
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
        prefix="model.layers.0.attn",
    )


@pytest.mark.push
def test_dsv4_oot_wrapper_registered():
    """Importing the plugin registers the OOT wrapper under the base class name
    that PluggableLayer.__new__ dispatches on."""
    import vllm_tt.attention_impls.attention_dsv4 as dsv4  # noqa: F401
    from vllm.model_executor.custom_op import op_registry_oot

    assert dsv4._DSV4_WRAPPER_AVAILABLE
    assert (
        op_registry_oot.get("DeepseekV4MultiHeadLatentAttentionWrapper")
        is dsv4.TTDeepseekV4MLAWrapper
    )


@pytest.mark.push
def test_dsv4_oot_wrapper_constructs_on_tt():
    """The CUDA-only upstream wrapper ctor runs to completion on tt under the
    monkeypatch recipe, producing our OOT subclass with its submodules built."""
    import vllm_tt.attention_impls.attention_dsv4 as dsv4  # noqa: F401

    cfg = _tiny_dsv4_config()
    vllm_config = _mock_vllm_config(cfg)
    orig_cache_dtype = vllm_config.cache_config.cache_dtype

    with _construction_context(vllm_config):
        wrapper = _build_wrapper(vllm_config)

    # Dispatched to our OOT subclass and fully constructed.
    assert type(wrapper).__name__ == "TTDeepseekV4MLAWrapper"
    assert isinstance(wrapper, dsv4.TTDeepseekV4MLAWrapper)
    assert hasattr(wrapper, "mla_attn")  # DeepseekV4MLAAttention built
    assert hasattr(wrapper, "swa_cache_layer")  # SWA cache built
    # The temporary fp8 cache-dtype patch was reverted (tt stays bf16/auto).
    assert vllm_config.cache_config.cache_dtype == orig_cache_dtype


@pytest.mark.push
def test_dsv4_oot_wrapper_forward_rejects_batched():
    """forward supports a single [tokens, hidden] prefill sequence; batched /
    paged-decode input (which needs the model_runner KV-cache + metadata
    plumbing) is explicitly rejected with NotImplementedError."""
    import vllm_tt.attention_impls.attention_dsv4 as dsv4  # noqa: F401

    cfg = _tiny_dsv4_config()
    vllm_config = _mock_vllm_config(cfg)
    with _construction_context(vllm_config):
        wrapper = _build_wrapper(vllm_config)
        with pytest.raises(NotImplementedError):
            wrapper.forward(
                torch.zeros(2, 4, dtype=torch.int32),  # 3D batched -> unsupported
                torch.zeros(2, 4, cfg.hidden_size, dtype=torch.bfloat16),
            )
