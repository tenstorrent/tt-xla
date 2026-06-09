# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pytest
import torch
import torch_xla
import torch_xla.runtime as xr
from infra import Framework, MLACache, run_graph_test
from infra.evaluators import ComparisonConfig, PccConfig
from torch import nn
from torch_xla.distributed.spmd import Mesh

from third_party.tt_forge_models.deepseek.deepseek_v3_1.pytorch.loader import (
    ModelLoader,
)


def _attention_shard_spec(attn):
    """Per-attention shard spec, mirrors ``loader.load_shard_spec`` for ``self_attn``."""
    return {
        attn.q_a_proj.weight: (None, "model"),
        attn.q_b_proj.weight: ("model", None),
        attn.kv_a_proj_with_mqa.weight: (None, "model"),
        attn.kv_b_proj.weight: ("model", None),
        attn.o_proj.weight: (None, "model"),
    }


def _moe_shard_spec(mlp_wrapper):
    """A2aSparseMLPWithSharedExperts shard spec, mirrors ``loader.load_shard_spec``."""
    shard_specs = {}
    inner = mlp_wrapper.mlp
    shard_specs[inner.router.gate.weight] = (None, "model")
    shard_specs[inner.experts.gate_proj] = (("model", "batch"), None, None)
    shard_specs[inner.experts.up_proj] = (("model", "batch"), None, None)
    shard_specs[inner.experts.down_proj] = (("model", "batch"), None, None)
    for bias_name in ("gate_proj_bias", "up_proj_bias", "down_proj_bias"):
        b = getattr(inner.experts, bias_name, None)
        if b is not None:
            shard_specs[b] = (("model", "batch"), None)
    shared = mlp_wrapper.shared_experts
    if shared is not None:
        shard_specs[shared.gate_proj.weight] = (None, "model")
        shard_specs[shared.up_proj.weight] = (None, "model")
        shard_specs[shared.down_proj.weight] = ("model", None)
    return shard_specs


def _decoder_layer_shard_spec(layer):
    """Full decoder-layer shard spec (attn + norms + mlp), mirrors ``loader.load_shard_spec``."""
    shard_specs = _attention_shard_spec(layer.self_attn)
    shard_specs[layer.input_layernorm.weight] = ("model",)
    shard_specs[layer.post_attention_layernorm.weight] = ("model",)
    # MoE layers have .mlp wrapped in A2aSparseMLPWithSharedExperts (after
    # enable_sparse_mlp); dense layers keep DeepseekV3MLP with gate/up/down_proj.
    if hasattr(layer.mlp, "shared_experts"):
        shard_specs.update(_moe_shard_spec(layer.mlp))
    else:
        shard_specs[layer.mlp.gate_proj.weight] = ("batch", "model")
        shard_specs[layer.mlp.up_proj.weight] = ("batch", "model")
        shard_specs[layer.mlp.down_proj.weight] = ("model", "batch")
    return shard_specs


def _make_mla_cache(config, batch_size, max_cache_len, dtype=torch.bfloat16):
    """Pre-allocated MLA cache on CPU; mirrors llm_utils.decode_utils.init_mla_cache."""
    cache = MLACache(config=config, max_cache_len=max_cache_len)
    dummy_kv = torch.zeros(
        (batch_size, 1, 1, config.kv_lora_rank), dtype=dtype, device="cpu"
    )
    dummy_pe = torch.zeros(
        (batch_size, 1, 1, config.qk_rope_head_dim), dtype=dtype, device="cpu"
    )
    for layer in cache.layers:
        layer.lazy_initialization(dummy_kv, dummy_pe)
    return cache


class _HeadWrapper(nn.Module):
    """Composition of the model's final RMSNorm and lm_head — the vocab projection
    chain the benchmark applies after the last decoder layer (see
    ``llm_benchmark.py``: ``model.lm_head(model.model.norm(...))`` via the
    LLMSamplingWrapper)."""

    def __init__(self, norm, lm_head):
        super().__init__()
        self.norm = norm
        self.lm_head = lm_head

    def forward(self, hidden_states):
        return self.lm_head(self.norm(hidden_states))


@pytest.mark.nightly
@pytest.mark.galaxy
@pytest.mark.parametrize("batch_size", [64])
@pytest.mark.parametrize("seq_len", [128])
def test_deepseek_v3_1_moe_block(batch_size, seq_len):
    """PCC test for the first MoE block of DeepSeek V3.1 with real weights.

    Loads the 4-layer model via the V3.1 ModelLoader (which downloads BF16
    weights and applies enable_sparse_mlp), then isolates the MoE at layer
    ``first_k_dense_replace`` (= 3) and runs it through run_graph_test on a
    Galaxy (4, 8) mesh.
    """
    xr.set_device_type("TT")
    torch_xla.runtime.use_spmd()

    loader = ModelLoader(num_layers=4)
    model = loader.load_model(dtype_override=torch.bfloat16)
    config = loader.config

    # first_k_dense_replace=3 -> layer 3 is the first MoE layer. After
    # enable_sparse_mlp the .mlp is A2aSparseMLPWithSharedExperts.
    moe_block = model.model.layers[config.first_k_dense_replace].mlp
    moe_block.eval()

    hidden_states = torch.randn(
        (batch_size, seq_len, config.hidden_size), dtype=torch.bfloat16
    )

    num_devices = xr.global_runtime_device_count()
    mesh_shape, axis_names = loader.get_mesh_config(num_devices)
    device_ids = np.array(range(num_devices))
    mesh = Mesh(device_ids, mesh_shape, axis_names)

    def get_shard_spec(mlp_wrapper, args, kwargs):
        shard_specs = _moe_shard_spec(mlp_wrapper)
        # hidden_states: [batch, seq, hidden] — batch on "batch" (4), hidden on "model" (8).
        shard_specs[args[0]] = ("batch", None, "model")
        return shard_specs

    comparison_config = ComparisonConfig(
        pcc=PccConfig(enabled=True, required_pcc=0.997),
    )

    run_graph_test(
        moe_block,
        [hidden_states],
        framework=Framework.TORCH,
        mesh=mesh,
        shard_spec_fn=get_shard_spec,
        comparison_config=comparison_config,
    )


@pytest.mark.nightly
@pytest.mark.galaxy
@pytest.mark.parametrize("batch_size", [64])
@pytest.mark.parametrize("seq_len", [128])
def test_deepseek_v3_1_attention_prefill(batch_size, seq_len):
    """PCC test for DeepSeek V3.1 attention (prefill) with real weights.

    Loads the 4-layer model, isolates ``self_attn`` from layer 0, and runs a
    prefill-style forward (full causal mask, ``cache_position=arange(seq_len)``)
    against an empty MLA cache. Matches the benchmark setup in
    ``tests/benchmark/test_llms.py::test_deepseek_v3_1_tp_galaxy_4_layers``
    (batch=64, input_sequence_length=128, ``use_mla_cache=True``).
    """
    xr.set_device_type("TT")
    torch_xla.runtime.use_spmd()

    loader = ModelLoader(num_layers=4)
    model = loader.load_model(dtype_override=torch.bfloat16)
    config = loader.config

    attention = model.model.layers[0].self_attn
    attention.eval()

    # Benchmark uses max_cache_len == input_sequence_length (see llm_benchmark.py).
    max_cache_len = seq_len

    hidden_states = torch.randn(
        (batch_size, seq_len, config.hidden_size), dtype=torch.bfloat16
    )
    # Additive 4D causal mask: 0 = attend, -inf = mask. Shape (B, 1, q, kv).
    attention_mask = torch.zeros(
        batch_size, 1, seq_len, max_cache_len, dtype=torch.bfloat16
    )
    attention_mask.masked_fill_(
        ~torch.ones(seq_len, max_cache_len).bool().tril(), float("-inf")
    )
    position_ids = torch.arange(seq_len).unsqueeze(0)
    cache_position = torch.arange(seq_len, dtype=torch.long)

    past_key_value = _make_mla_cache(config, batch_size, max_cache_len)

    num_devices = xr.global_runtime_device_count()
    mesh_shape, axis_names = loader.get_mesh_config(num_devices)
    device_ids = np.array(range(num_devices))
    mesh = Mesh(device_ids, mesh_shape, axis_names)

    def get_shard_spec(attn, args, kwargs):
        shard_specs = _attention_shard_spec(attn)
        # hidden_states (B, S, H) — batch on "batch", hidden on "model".
        shard_specs[args[0]] = ("batch", None, "model")
        # attention_mask (B, 1, q, kv) — batch on "batch".
        shard_specs[args[1]] = ("batch", None, None, None)
        # MLA cache: compressed_kv / k_pe (B, 1, max_cache, dim) — batch on "batch".
        cache = args[3]
        shard_specs[cache.layers[0].compressed_kv] = ("batch", None, None, None)
        shard_specs[cache.layers[0].k_pe] = ("batch", None, None, None)
        return shard_specs

    comparison_config = ComparisonConfig(
        pcc=PccConfig(enabled=True, required_pcc=0.998),
    )

    run_graph_test(
        attention,
        [
            hidden_states,
            attention_mask,
            position_ids,
            past_key_value,
            False,  # output_attentions
            True,  # use_cache
            cache_position,
        ],
        framework=Framework.TORCH,
        mesh=mesh,
        shard_spec_fn=get_shard_spec,
        comparison_config=comparison_config,
    )


@pytest.mark.nightly
@pytest.mark.galaxy
@pytest.mark.parametrize("batch_size", [64])
@pytest.mark.parametrize("prefill_len", [127])
def test_deepseek_v3_1_attention_decode(batch_size, prefill_len):
    """PCC test for DeepSeek V3.1 attention (decode step) with real weights.

    Decodes a single token at ``cache_position=[prefill_len]`` against an
    MLA cache pre-populated by running the same attention on CPU for a
    ``prefill_len``-token prompt. Matches the post-prefill state of the
    benchmark (batch=64, max_cache_len=128 from input_sequence_length=128).
    """
    xr.set_device_type("TT")
    torch_xla.runtime.use_spmd()

    loader = ModelLoader(num_layers=4)
    model = loader.load_model(dtype_override=torch.bfloat16)
    config = loader.config

    attention = model.model.layers[0].self_attn
    attention.eval()

    seq_len = 1
    # Pinned to benchmark default (input_sequence_length=128); prefill_len must
    # leave at least one decode slot, so prefill_len < max_cache_len.
    max_cache_len = 128

    past_key_value = _make_mla_cache(config, batch_size, max_cache_len)

    # Warm the cache on CPU so decode sees a realistic post-prefill state. Both
    # the CPU golden run and the TT run will get an independent ``copy.copy``
    # of this populated cache (see torch_device_runner.to_device).
    prefill_hidden = torch.randn(
        (batch_size, prefill_len, config.hidden_size), dtype=torch.bfloat16
    )
    prefill_mask = torch.zeros(
        batch_size, 1, prefill_len, max_cache_len, dtype=torch.bfloat16
    )
    prefill_mask.masked_fill_(
        ~torch.ones(prefill_len, max_cache_len).bool().tril(), float("-inf")
    )
    with torch.no_grad():
        attention(
            prefill_hidden,
            prefill_mask,
            torch.arange(prefill_len).unsqueeze(0),
            past_key_value=past_key_value,
            output_attentions=False,
            use_cache=True,
            cache_position=torch.arange(prefill_len, dtype=torch.long),
        )

    # Decode-step inputs: single new token at position ``prefill_len``.
    hidden_states = torch.randn(
        (batch_size, seq_len, config.hidden_size), dtype=torch.bfloat16
    )
    # Decode mask: attend to positions [0, prefill_len], mask out the rest.
    key_idx = torch.arange(max_cache_len)
    mask_row = torch.where(
        key_idx > prefill_len,
        torch.tensor(float("-inf"), dtype=torch.bfloat16),
        torch.tensor(0.0, dtype=torch.bfloat16),
    )
    attention_mask = (
        mask_row.view(1, 1, 1, max_cache_len)
        .expand(batch_size, 1, 1, max_cache_len)
        .contiguous()
    )
    position_ids = torch.tensor([[prefill_len]], dtype=torch.long)
    cache_position = torch.tensor([prefill_len], dtype=torch.long)

    num_devices = xr.global_runtime_device_count()
    mesh_shape, axis_names = loader.get_mesh_config(num_devices)
    device_ids = np.array(range(num_devices))
    mesh = Mesh(device_ids, mesh_shape, axis_names)

    def get_shard_spec(attn, args, kwargs):
        shard_specs = _attention_shard_spec(attn)
        shard_specs[args[0]] = ("batch", None, "model")
        shard_specs[args[1]] = ("batch", None, None, None)
        cache = args[3]
        shard_specs[cache.layers[0].compressed_kv] = ("batch", None, None, None)
        shard_specs[cache.layers[0].k_pe] = ("batch", None, None, None)
        return shard_specs

    comparison_config = ComparisonConfig(
        pcc=PccConfig(enabled=True, required_pcc=0.998),
    )

    run_graph_test(
        attention,
        [
            hidden_states,
            attention_mask,
            position_ids,
            past_key_value,
            False,  # output_attentions
            True,  # use_cache
            cache_position,
        ],
        framework=Framework.TORCH,
        mesh=mesh,
        shard_spec_fn=get_shard_spec,
        comparison_config=comparison_config,
    )


@pytest.mark.nightly
@pytest.mark.galaxy
@pytest.mark.parametrize("batch_size", [64])
@pytest.mark.parametrize("seq_len", [128])
@pytest.mark.parametrize(
    "layer_idx",
    [pytest.param(0, id="dense"), pytest.param(3, id="moe")],
)
def test_deepseek_v3_1_decoder_layer_prefill(batch_size, seq_len, layer_idx):
    """PCC test for one full DeepSeek V3.1 decoder layer (prefill).

    Layer 0 has a dense MLP; layer 3 is the first MoE layer
    (``first_k_dense_replace=3``). Exposes the bf16 residual-stream drift and
    the input/post-attention layernorms that the attention/MoE block tests
    don't cover, which is where the per-layer PCC gap likely comes from.
    """
    xr.set_device_type("TT")
    torch_xla.runtime.use_spmd()

    loader = ModelLoader(num_layers=4)
    model = loader.load_model(dtype_override=torch.bfloat16)
    config = loader.config

    decoder_layer = model.model.layers[layer_idx]
    decoder_layer.eval()

    max_cache_len = seq_len
    hidden_states = torch.randn(
        (batch_size, seq_len, config.hidden_size), dtype=torch.bfloat16
    )
    attention_mask = torch.zeros(
        batch_size, 1, seq_len, max_cache_len, dtype=torch.bfloat16
    )
    attention_mask.masked_fill_(
        ~torch.ones(seq_len, max_cache_len).bool().tril(), float("-inf")
    )
    position_ids = torch.arange(seq_len).unsqueeze(0)
    cache_position = torch.arange(seq_len, dtype=torch.long)

    past_key_value = _make_mla_cache(config, batch_size, max_cache_len)

    num_devices = xr.global_runtime_device_count()
    mesh_shape, axis_names = loader.get_mesh_config(num_devices)
    device_ids = np.array(range(num_devices))
    mesh = Mesh(device_ids, mesh_shape, axis_names)

    def get_shard_spec(layer, args, kwargs):
        shard_specs = _decoder_layer_shard_spec(layer)
        shard_specs[args[0]] = ("batch", None, "model")
        shard_specs[args[1]] = ("batch", None, None, None)
        cache = args[3]
        shard_specs[cache.layers[layer_idx].compressed_kv] = (
            "batch",
            None,
            None,
            None,
        )
        shard_specs[cache.layers[layer_idx].k_pe] = ("batch", None, None, None)
        return shard_specs

    comparison_config = ComparisonConfig(
        pcc=PccConfig(enabled=True, required_pcc=0.998),
    )

    run_graph_test(
        decoder_layer,
        [
            hidden_states,
            attention_mask,
            position_ids,
            past_key_value,
            False,  # output_attentions
            True,  # use_cache
            cache_position,
        ],
        framework=Framework.TORCH,
        mesh=mesh,
        shard_spec_fn=get_shard_spec,
        comparison_config=comparison_config,
    )


@pytest.mark.nightly
@pytest.mark.galaxy
@pytest.mark.parametrize("batch_size", [64])
@pytest.mark.parametrize("prefill_len", [127])
@pytest.mark.parametrize(
    "layer_idx",
    [pytest.param(0, id="dense"), pytest.param(3, id="moe")],
)
def test_deepseek_v3_1_decoder_layer_decode(batch_size, prefill_len, layer_idx):
    """PCC test for one full DeepSeek V3.1 decoder layer (decode step).

    Warms layer ``layer_idx``'s MLA cache on CPU via a ``prefill_len``-token
    prefill, then decodes a single token at ``cache_position=[prefill_len]``.
    ``max_cache_len`` is pinned to the benchmark default (128).
    """
    xr.set_device_type("TT")
    torch_xla.runtime.use_spmd()

    loader = ModelLoader(num_layers=4)
    model = loader.load_model(dtype_override=torch.bfloat16)
    config = loader.config

    decoder_layer = model.model.layers[layer_idx]
    decoder_layer.eval()

    seq_len = 1
    max_cache_len = 128

    past_key_value = _make_mla_cache(config, batch_size, max_cache_len)

    # Warm the MLA cache on CPU with a real prefill through the layer; both the
    # CPU golden run and the TT run get an independent copy.copy of this
    # populated cache (see torch_device_runner.to_device).
    prefill_hidden = torch.randn(
        (batch_size, prefill_len, config.hidden_size), dtype=torch.bfloat16
    )
    prefill_mask = torch.zeros(
        batch_size, 1, prefill_len, max_cache_len, dtype=torch.bfloat16
    )
    prefill_mask.masked_fill_(
        ~torch.ones(prefill_len, max_cache_len).bool().tril(), float("-inf")
    )
    with torch.no_grad():
        decoder_layer(
            prefill_hidden,
            prefill_mask,
            torch.arange(prefill_len).unsqueeze(0),
            past_key_value=past_key_value,
            output_attentions=False,
            use_cache=True,
            cache_position=torch.arange(prefill_len, dtype=torch.long),
        )

    hidden_states = torch.randn(
        (batch_size, seq_len, config.hidden_size), dtype=torch.bfloat16
    )
    key_idx = torch.arange(max_cache_len)
    mask_row = torch.where(
        key_idx > prefill_len,
        torch.tensor(float("-inf"), dtype=torch.bfloat16),
        torch.tensor(0.0, dtype=torch.bfloat16),
    )
    attention_mask = (
        mask_row.view(1, 1, 1, max_cache_len)
        .expand(batch_size, 1, 1, max_cache_len)
        .contiguous()
    )
    position_ids = torch.tensor([[prefill_len]], dtype=torch.long)
    cache_position = torch.tensor([prefill_len], dtype=torch.long)

    num_devices = xr.global_runtime_device_count()
    mesh_shape, axis_names = loader.get_mesh_config(num_devices)
    device_ids = np.array(range(num_devices))
    mesh = Mesh(device_ids, mesh_shape, axis_names)

    def get_shard_spec(layer, args, kwargs):
        shard_specs = _decoder_layer_shard_spec(layer)
        shard_specs[args[0]] = ("batch", None, "model")
        shard_specs[args[1]] = ("batch", None, None, None)
        cache = args[3]
        shard_specs[cache.layers[layer_idx].compressed_kv] = (
            "batch",
            None,
            None,
            None,
        )
        shard_specs[cache.layers[layer_idx].k_pe] = ("batch", None, None, None)
        return shard_specs

    comparison_config = ComparisonConfig(
        pcc=PccConfig(enabled=True, required_pcc=0.998),
    )

    run_graph_test(
        decoder_layer,
        [
            hidden_states,
            attention_mask,
            position_ids,
            past_key_value,
            False,  # output_attentions
            True,  # use_cache
            cache_position,
        ],
        framework=Framework.TORCH,
        mesh=mesh,
        shard_spec_fn=get_shard_spec,
        comparison_config=comparison_config,
    )


@pytest.mark.nightly
@pytest.mark.galaxy
@pytest.mark.parametrize("batch_size", [64])
@pytest.mark.parametrize("seq_len", [128])
def test_deepseek_v3_1_head(batch_size, seq_len):
    """PCC test for the model head (final RMSNorm + lm_head) with real weights.

    Exercises the vocab projection over the residual stream — the part that
    sees the most accumulated bf16 drift in the benchmark and isn't covered
    by any per-block test.
    """
    xr.set_device_type("TT")
    torch_xla.runtime.use_spmd()

    loader = ModelLoader(num_layers=4)
    model = loader.load_model(dtype_override=torch.bfloat16)
    config = loader.config

    head = _HeadWrapper(model.model.norm, model.lm_head)
    head.eval()

    hidden_states = torch.randn(
        (batch_size, seq_len, config.hidden_size), dtype=torch.bfloat16
    )

    num_devices = xr.global_runtime_device_count()
    mesh_shape, axis_names = loader.get_mesh_config(num_devices)
    device_ids = np.array(range(num_devices))
    mesh = Mesh(device_ids, mesh_shape, axis_names)

    def get_shard_spec(head, args, kwargs):
        return {
            args[0]: ("batch", None, "model"),
            head.norm.weight: ("model",),
            # loader.load_shard_spec: lm_head.weight = (None, "model").
            head.lm_head.weight: (None, "model"),
        }

    comparison_config = ComparisonConfig(
        pcc=PccConfig(enabled=True, required_pcc=0.998),
    )

    run_graph_test(
        head,
        [hidden_states],
        framework=Framework.TORCH,
        mesh=mesh,
        shard_spec_fn=get_shard_spec,
        comparison_config=comparison_config,
    )
