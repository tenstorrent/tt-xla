# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import os

import numpy as np
import pytest
import torch
import torch_xla
import torch_xla.distributed.spmd as xs
import torch_xla.runtime as xr
from infra import Framework, run_graph_test
from infra.evaluators import ComparisonConfig, PccConfig
from torch_xla.distributed.spmd import Mesh
from transformers import DynamicCache
from tt_torch.sparse_mlp import create_a2a_from_deepseek_v3_moe, enable_sparse_mlp

from tests.utils import failed_ttmlir_compilation

from .configuration_deepseek import DeepseekV3Config
from .modeling_deepseek import (
    DeepseekV3Attention,
    DeepseekV3DecoderLayer,
    DeepseekV3ForCausalLM,
    DeepseekV3Model,
    DeepseekV3MoE,
)
from .original_modeling_deepseek import DeepseekV3Attention as OrigDeepseekV3Attention
from .utils import MLACache


@pytest.mark.xfail(
    reason=failed_ttmlir_compilation(
        "'ttir.concat' op Output tensor dimension 0 does not match the sum of input tensor dimensions: 1 vs. 32. "
    )
)
def test_kimi_k2_single_layer():
    xr.set_device_type("TT")

    # Load full Kimi K2 config from JSON file
    config_path = os.path.join(os.path.dirname(__file__), "config.json")
    config = DeepseekV3Config.from_json_file(config_path)

    # Override for single layer testing
    config.num_hidden_layers = 1
    config.use_cache = False

    model = DeepseekV3ForCausalLM(config)

    batch_size = 64
    seq_len = 32
    tokens = torch.randint(0, config.vocab_size, (batch_size, seq_len))
    model = model.to(torch.bfloat16)
    model = model.eval()

    compiled_model = torch.compile(model, backend="tt")

    device = torch_xla.device()
    tokens = tokens.to(device)
    compiled_model = compiled_model.to(device)

    with torch.no_grad():
        output = compiled_model(tokens)
        output.to("cpu")


@pytest.mark.nightly
@pytest.mark.llmbox
def test_kimi_k2_attention_prefill():
    xr.set_device_type("TT")
    torch_xla.runtime.use_spmd()

    # Load full Kimi K2 config from JSON file
    config_path = os.path.join(os.path.dirname(__file__), "config.json")
    config = DeepseekV3Config.from_json_file(config_path)

    attention = DeepseekV3Attention(config, layer_idx=0)
    attention = attention.to(torch.bfloat16)

    batch_size = 64
    seq_len = 32
    max_cache_len = 1024
    hidden_states = torch.randn(
        (batch_size, seq_len, config.hidden_size), dtype=torch.bfloat16
    )
    attention_mask = torch.rand(
        batch_size, 1, seq_len, max_cache_len, dtype=torch.bfloat16
    )

    num_devices = xr.global_runtime_device_count()
    mesh_shape = (2, 4)
    device_ids = np.array(range(num_devices))
    mesh = Mesh(device_ids, mesh_shape, ("batch", "model"))
    static_cache = MLACache(
        config=config,
        max_batch_size=batch_size,
        max_cache_len=max_cache_len,
        device="cpu",
        dtype=torch.bfloat16,
    )
    past_key_states = static_cache
    cache_positions = torch.randint(0, max_cache_len, (seq_len,), dtype=torch.long)
    position_ids = torch.arange(seq_len).unsqueeze(0)

    def get_shard_spec(attention, args, kwargs):
        shard_specs = {}

        shard_specs[args[0]] = (None, None, "batch")
        shard_specs[attention.q_b_proj.weight] = ("model", None)
        shard_specs[attention.kv_b_proj.weight] = ("model", None)
        shard_specs[attention.o_proj.weight] = ("batch", "model")

        # Consume hidden states, TP on batch dimension
        shard_specs[attention.q_a_proj.weight] = (None, "batch")
        shard_specs[attention.kv_a_proj_with_mqa.weight] = (None, "batch")
        return shard_specs

    comparison_config = ComparisonConfig(
        pcc=PccConfig(enabled=True, required_pcc=0.95),
    )

    run_graph_test(
        attention,
        [
            hidden_states,
            attention_mask,
            position_ids,
            past_key_states,
            False,
            True,
            cache_positions,
        ],
        framework=Framework.TORCH,
        mesh=mesh,
        shard_spec_fn=get_shard_spec,
        comparison_config=comparison_config,
    )


@pytest.mark.nightly
@pytest.mark.llmbox
def test_kimi_k2_attention_decode():
    xr.set_device_type("TT")
    torch_xla.runtime.use_spmd()

    config_path = os.path.join(os.path.dirname(__file__), "config.json")
    config = DeepseekV3Config.from_json_file(config_path)
    config.num_hidden_layers = 1

    attention = DeepseekV3Attention(config, layer_idx=0)
    attention = attention.to(torch.bfloat16)

    max_cache_len = 1024
    batch_size = 64
    seq_len = 1
    hidden_states = torch.randn(
        (batch_size, seq_len, config.hidden_size), dtype=torch.bfloat16
    )
    attention_mask = torch.rand(
        batch_size, 1, seq_len, max_cache_len, dtype=torch.bfloat16
    )

    num_devices = xr.global_runtime_device_count()
    mesh_shape = (2, 4)
    device_ids = np.array(range(num_devices))
    mesh = Mesh(device_ids, mesh_shape, ("_axis_0", "_axis_1"))

    position_ids = torch.arange(seq_len).unsqueeze(0)
    cache_positions = torch.randint(0, max_cache_len, (seq_len,), dtype=torch.long)
    static_cache = MLACache(
        config=config,
        max_batch_size=batch_size,
        max_cache_len=max_cache_len,
        device="cpu",
        dtype=torch.bfloat16,
    )
    past_key_states = static_cache

    def get_shard_spec(attention, args, kwargs):
        shard_specs = {}

        shard_specs[args[0]] = ("_axis_1", None, "_axis_0")
        shard_specs[args[1]] = ("_axis_1", None, None, None)
        shard_specs[args[3][0][0]] = ("_axis_1", None, None, None)
        shard_specs[args[3][0][1]] = ("_axis_1", None, None, None)

        # Main attention weights, TP across model and batch dimensions
        shard_specs[attention.q_b_proj.weight] = ("_axis_0", None)
        shard_specs[attention.kv_b_proj.weight] = ("_axis_0", None)
        shard_specs[attention.o_proj.weight] = (None, "_axis_0")

        # Consume hidden states, TP on batch dimension
        shard_specs[attention.q_a_proj.weight] = (None, "_axis_0")
        shard_specs[attention.kv_a_proj_with_mqa.weight] = (None, "_axis_0")
        return shard_specs

    run_graph_test(
        attention,
        [
            hidden_states,
            attention_mask,
            position_ids,
            past_key_states,
            False,
            True,
            cache_positions,
        ],
        framework=Framework.TORCH,
        mesh=mesh,
        shard_spec_fn=get_shard_spec,
    )


@pytest.mark.nightly
@pytest.mark.llmbox
def test_kimi_k2_layer():
    xr.set_device_type("TT")
    torch_xla.runtime.use_spmd()

    # Load full Kimi K2 config from JSON file
    config_path = os.path.join(os.path.dirname(__file__), "config.json")
    config = DeepseekV3Config.from_json_file(config_path)
    config._attn_implementation = "eager"
    config.num_hidden_layers = 1

    layer = DeepseekV3DecoderLayer(config, layer_idx=0)
    layer = layer.to(torch.bfloat16)

    max_cache_len = 1024
    batch_size = 64
    seq_len = 1
    hidden_states = torch.randn(
        (batch_size, seq_len, config.hidden_size), dtype=torch.bfloat16
    )
    attention_mask = torch.rand(
        batch_size, 1, seq_len, max_cache_len, dtype=torch.bfloat16
    )
    cache_positions = torch.randint(0, max_cache_len, (seq_len,), dtype=torch.long)
    num_devices = xr.global_runtime_device_count()
    mesh_shape = (2, 4)
    device_ids = np.array(range(num_devices))
    mesh = Mesh(device_ids, mesh_shape, ("_axis_0", "_axis_1"))

    position_ids = torch.arange(seq_len).unsqueeze(0)
    static_cache = MLACache(
        config=config,
        max_batch_size=batch_size,
        max_cache_len=max_cache_len,
        device="cpu",
        dtype=torch.bfloat16,
    )
    past_key_states = static_cache

    num_devices = xr.global_runtime_device_count()
    device_ids = np.array(range(num_devices))

    def get_shard_spec(layer, args, kwargs):
        shard_specs = {}

        shard_specs[args[0]] = ("_axis_1", None, "_axis_0")
        shard_specs[args[1]] = ("_axis_1", None, None, None)
        shard_specs[args[3][0][0]] = ("_axis_1", None, None, None)
        shard_specs[args[3][0][1]] = ("_axis_1", None, None, None)

        # Main attention weights, TP across model and batch dimensions
        shard_specs[layer.self_attn.q_b_proj.weight] = ("_axis_0", None)
        shard_specs[layer.self_attn.kv_b_proj.weight] = ("_axis_0", None)
        shard_specs[layer.self_attn.o_proj.weight] = (None, "_axis_0")

        # Consume hidden states, TP on batch dimension
        shard_specs[layer.self_attn.q_a_proj.weight] = (None, "_axis_0")
        shard_specs[layer.self_attn.kv_a_proj_with_mqa.weight] = (None, "_axis_0")

        shard_specs[layer.mlp.gate_proj.weight] = ("_axis_1", "_axis_0")
        shard_specs[layer.mlp.up_proj.weight] = ("_axis_1", "_axis_0")
        shard_specs[layer.mlp.down_proj.weight] = ("_axis_0", "_axis_1")

        shard_specs[layer.input_layernorm.weight] = ("_axis_0",)
        shard_specs[layer.post_attention_layernorm.weight] = ("_axis_0",)

        return shard_specs

    run_graph_test(
        layer,
        [
            hidden_states,
            attention_mask,
            position_ids,
            past_key_states,
            False,
            True,
            cache_positions,
        ],
        framework=Framework.TORCH,
        mesh=mesh,
        shard_spec_fn=get_shard_spec,
    )


@pytest.mark.nightly
@pytest.mark.llmbox
def test_kimi_k2_layer_sparse_moe():
    xr.set_device_type("TT")
    torch_xla.runtime.use_spmd()

    # Load full Kimi K2 config from JSON file
    config_path = os.path.join(os.path.dirname(__file__), "config.json")
    config = DeepseekV3Config.from_json_file(config_path)
    config._attn_implementation = "eager"
    config.num_hidden_layers = 2

    layer = DeepseekV3DecoderLayer(config, layer_idx=1)
    layer = layer.eval().to(torch.bfloat16)

    max_cache_len = 1024
    batch_size = 64
    seq_len = 1
    hidden_states = torch.randn(
        (batch_size, seq_len, config.hidden_size), dtype=torch.bfloat16
    )
    attention_mask = torch.rand(
        batch_size, 1, seq_len, max_cache_len, dtype=torch.bfloat16
    )
    cache_positions = torch.randint(0, max_cache_len, (seq_len,), dtype=torch.long)
    num_devices = xr.global_runtime_device_count()
    mesh_shape = (2, 4)
    device_ids = np.array(range(num_devices))
    mesh = Mesh(device_ids, mesh_shape, ("_axis_0", "_axis_1"))
    enable_sparse_mlp(layer, mesh=mesh_shape, cluster_axis=0, config=config)

    position_ids = torch.arange(seq_len).unsqueeze(0)
    static_cache = MLACache(
        config=config,
        max_batch_size=batch_size,
        max_cache_len=max_cache_len,
        device="cpu",
        dtype=torch.bfloat16,
    )
    past_key_states = static_cache

    num_devices = xr.global_runtime_device_count()
    device_ids = np.array(range(num_devices))

    def get_shard_spec(layer, args, kwargs):
        shard_specs = {}

        # Activations: fully replicated (no batch/hidden sharding).
        # all_to_all operates on axis_0, so batch must NOT be sharded on axis_0
        # (otherwise: all_gather before dispatch + mesh_partition after combine).
        # all_reduce operates on axis_1, so batch must NOT be sharded on axis_1
        # (otherwise: all_reduce would mix different tokens across axis_1).
        shard_specs[args[0]] = ("_axis_1", None, "_axis_0")
        shard_specs[args[1]] = ("_axis_1", None, None, None)
        shard_specs[args[3][1][0]] = ("_axis_1", None, None, None)
        shard_specs[args[3][1][1]] = ("_axis_1", None, None, None)

        # Attention: TP on axis_1 for head parallelism
        shard_specs[layer.self_attn.q_b_proj.weight] = ("_axis_0", None)
        shard_specs[layer.self_attn.kv_b_proj.weight] = ("_axis_0", None)
        shard_specs[layer.self_attn.o_proj.weight] = (None, "_axis_0")

        # q_a/kv_a projections: hidden is replicated, small output dims
        shard_specs[layer.self_attn.q_a_proj.weight] = (None, "_axis_0")
        shard_specs[layer.self_attn.kv_a_proj_with_mqa.weight] = (None, "_axis_0")

        # A2aSparseMLP: experts compound-sharded (axis_0, axis_1)
        mlp_wrapper = layer.mlp
        mlp = mlp_wrapper.mlp if hasattr(mlp_wrapper, "mlp") else mlp_wrapper
        shard_specs[mlp.router.gate.weight] = (None, "_axis_0")
        shard_specs[mlp.experts.gate_up_proj] = (
            ("_axis_0", "_axis_1"),
            None,
            None,
        )
        shard_specs[mlp.experts.down_proj] = (
            ("_axis_0", "_axis_1"),
            None,
            None,
        )
        shard_specs[mlp.experts.gate_up_proj_bias] = (("_axis_0", "_axis_1"), None)
        shard_specs[mlp.experts.down_proj_bias] = (("_axis_0", "_axis_1"), None)

        # Shared experts (if present, on wrapper not on inner A2aSparseMLP)
        shared = getattr(mlp_wrapper, "shared_experts", None)
        if shared is not None:
            shard_specs[shared.gate_proj.weight] = (None, "_axis_0")
            shard_specs[shared.up_proj.weight] = (None, "_axis_0")
            shard_specs[shared.down_proj.weight] = ("_axis_0", None)

        # Layernorm: hidden replicated → weight not sharded
        shard_specs[layer.input_layernorm.weight] = ("_axis_0",)
        shard_specs[layer.post_attention_layernorm.weight] = ("_axis_0",)

        return shard_specs

    run_graph_test(
        layer,
        [
            hidden_states,
            attention_mask,
            position_ids,
            past_key_states,
            False,
            True,
            cache_positions,
        ],
        framework=Framework.TORCH,
        mesh=mesh,
        shard_spec_fn=get_shard_spec,
    )


def test_kimi_k2_full():
    """Test full Kimi K2 model (DeepseekV3Model) with A2aSparseMLP on (2,4) mesh."""

    xr.set_device_type("TT")
    torch_xla.runtime.use_spmd()

    num_devices_total = xr.global_runtime_device_count()

    # Load full Kimi K2 config from JSON file
    config_path = os.path.join(os.path.dirname(__file__), "config.json")
    config = DeepseekV3Config.from_json_file(config_path)
    config._attn_implementation = "eager"
    if num_devices_total == 8:
        config.num_hidden_layers = 2  # Need 2+ so layer_idx=1 exists and is MoE
    use_cache = True
    config.use_cache = use_cache

    model = DeepseekV3Model(config)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"[JAMES] Total model parameters for kimi k2 models {total_params:,} with 2 layers")
    model = model.to(torch.bfloat16)

    # Replace all MoE MLP layers with A2aSparseMLP
    # With first_k_dense_replace=1 and moe_layer_freq=1:
    #   Layer 0: Dense MLP (layer_idx < first_k_dense_replace), skip
    #   Layer 1+: MoE, replace with A2aSparseMLP
    mesh_shape = (2, 4)
    if num_devices_total == 32:
        mesh_shape = (8, 4)
    elif num_devices_total == 64:
        mesh_shape = (8, 8)

    enable_sparse_mlp(
        model,
        mesh=mesh_shape,
        cluster_axis=0,
        config=config,
    )

    model.eval()

    max_cache_len = 1024
    batch_size = 64
    seq_len = 1
    input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len))
    attention_mask = torch.rand(batch_size, 1, seq_len, max_cache_len, dtype=torch.bfloat16)
    cache_positions = torch.randint(0, max_cache_len, (seq_len,), dtype=torch.long)
    position_ids = torch.arange(seq_len).unsqueeze(0)
    static_cache = MLACache(
        config=config,
        max_batch_size=batch_size,
        max_cache_len=max_cache_len,
        device="cpu",
        dtype=torch.bfloat16,
    )

    device_ids = np.array(range(num_devices_total))
    mesh = Mesh(device_ids, mesh_shape, ("_axis_0", "_axis_1"))

    def get_shard_spec(model, args, kwargs):
        shard_specs = {}
        shard_specs[args[0]] = ("_axis_1", None)
        shard_specs[args[1]] = ("_axis_1", None, None, None)

        # Cache shard specs per layer
        for layer_idx in range(config.num_hidden_layers):
            shard_specs[args[3][layer_idx][0]] = ("_axis_1", None, None, None)
            shard_specs[args[3][layer_idx][1]] = ("_axis_1", None, None, None)

        # Embedding
        shard_specs[model.embed_tokens.weight] = (None, "_axis_0")

        # Per-layer sharding
        for decoder_layer in model.layers:
            # Attention weights
            shard_specs[decoder_layer.self_attn.q_b_proj.weight] = ("_axis_0", None)
            shard_specs[decoder_layer.self_attn.kv_b_proj.weight] = ("_axis_0", None)
            shard_specs[decoder_layer.self_attn.o_proj.weight] = (None, "_axis_0")
            shard_specs[decoder_layer.self_attn.q_a_proj.weight] = (None, "_axis_0")
            shard_specs[decoder_layer.self_attn.kv_a_proj_with_mqa.weight] = (
                None,
                "_axis_0",
            )

            # MLP sharding (MoE or dense)
            mlp_wrapper = decoder_layer.mlp
            if hasattr(mlp_wrapper, "mlp"):
                # A2aSparseMLP: experts compound-sharded (axis_0, axis_1)
                mlp = mlp_wrapper.mlp
                shard_specs[mlp.router.gate.weight] = (None, "_axis_0")
                shard_specs[mlp.experts.gate_up_proj] = (
                    ("_axis_0", "_axis_1"),
                    None,
                    None,
                )
                shard_specs[mlp.experts.down_proj] = (
                    ("_axis_0", "_axis_1"),
                    None,
                    None,
                )
                shard_specs[mlp.experts.gate_up_proj_bias] = (
                    ("_axis_0", "_axis_1"),
                    None,
                )
                shard_specs[mlp.experts.down_proj_bias] = (("_axis_0", "_axis_1"), None)

                # Shared experts
                shared = getattr(mlp_wrapper, "shared_experts", None)
                if shared is not None:
                    shard_specs[shared.gate_proj.weight] = (None, "_axis_0")
                    shard_specs[shared.up_proj.weight] = (None, "_axis_0")
                    shard_specs[shared.down_proj.weight] = ("_axis_0", None)
            else:
                # Dense MLP
                shard_specs[mlp_wrapper.gate_proj.weight] = ("_axis_1", "_axis_0")
                shard_specs[mlp_wrapper.up_proj.weight] = ("_axis_1", "_axis_0")
                shard_specs[mlp_wrapper.down_proj.weight] = ("_axis_0", "_axis_1")

            # LayerNorm
            shard_specs[decoder_layer.input_layernorm.weight] = ("_axis_0",)
            shard_specs[decoder_layer.post_attention_layernorm.weight] = ("_axis_0",)

        # Final norm
        shard_specs[model.norm.weight] = ("_axis_0",)

        return shard_specs

    comparison_config = ComparisonConfig(
        pcc=PccConfig(enabled=True, required_pcc=0.99),
    )

    run_graph_test(
        model,
        [
            input_ids,
            attention_mask,
            position_ids,
            static_cache,
            None,
            cache_positions,
            use_cache,
        ],
        framework=Framework.TORCH,
        mesh=mesh,
        shard_spec_fn=get_shard_spec,
        comparison_config=comparison_config,
    )


@pytest.mark.nightly
def test_kimi_k2_mla_cache():
    """
    CPU-only test validating the MLACache used in modeling_deepseek.py against the original
    DynamicCache used in original_modeling_deepseek.py for DeepseekV3Attention.
    """

    config = DeepseekV3Config(
        hidden_size=64,
        num_attention_heads=4,
        q_lora_rank=32,
        kv_lora_rank=16,
        qk_rope_head_dim=16,
        v_head_dim=8,
        qk_nope_head_dim=8,
        num_hidden_layers=1,
        max_position_embeddings=64,
        rope_scaling=None,
        attention_bias=False,
        attention_dropout=0.0,
    )

    BATCH_SIZE = 2
    PREFILL_LEN = 8
    LAYER_IDX = 0
    MAX_CACHE_LEN = PREFILL_LEN + 1

    mla_attn = DeepseekV3Attention(config, layer_idx=LAYER_IDX)
    mla_attn.eval()
    orig_attn = OrigDeepseekV3Attention(config, layer_idx=LAYER_IDX)
    orig_attn.load_state_dict(mla_attn.state_dict())
    orig_attn.eval()

    torch.manual_seed(0)

    # Prefill
    mla_cache = MLACache(config, max_cache_len=MAX_CACHE_LEN)
    orig_cache = DynamicCache()

    prefill_hidden = torch.randn(BATCH_SIZE, PREFILL_LEN, config.hidden_size)
    prefill_position_ids = torch.arange(PREFILL_LEN).unsqueeze(0).expand(BATCH_SIZE, -1)

    mla_prefill_mask = torch.zeros(BATCH_SIZE, 1, PREFILL_LEN, MAX_CACHE_LEN)
    orig_prefill_mask = torch.zeros(BATCH_SIZE, 1, PREFILL_LEN, PREFILL_LEN)

    with torch.no_grad():
        mla_attn(
            prefill_hidden,
            mla_prefill_mask,
            prefill_position_ids,
            past_key_value=mla_cache,
            use_cache=True,
            cache_position=torch.arange(PREFILL_LEN),
        )
        orig_attn(
            prefill_hidden,
            orig_prefill_mask,
            prefill_position_ids,
            past_key_value=orig_cache,
            use_cache=True,
        )

    # Decode
    decode_hidden = torch.randn(BATCH_SIZE, 1, config.hidden_size)
    decode_position_ids = torch.full((BATCH_SIZE, 1), PREFILL_LEN, dtype=torch.long)
    decode_mask = torch.zeros(BATCH_SIZE, 1, 1, MAX_CACHE_LEN)

    with torch.no_grad():
        mla_attn(
            decode_hidden,
            decode_mask,
            decode_position_ids,
            past_key_value=mla_cache,
            use_cache=True,
            cache_position=torch.tensor([PREFILL_LEN]),
        )
        orig_attn(
            decode_hidden,
            decode_mask,
            decode_position_ids,
            past_key_value=orig_cache,
            use_cache=True,
        )

    total_len = PREFILL_LEN + 1

    orig_key = orig_cache.layers[LAYER_IDX].keys
    orig_val = orig_cache.layers[LAYER_IDX].values

    compressed_kv = mla_cache.layers[LAYER_IDX].compressed_kv[:, 0, :total_len, :]
    mla_k_pe = mla_cache.layers[LAYER_IDX].k_pe[:, :, :total_len, :]

    with torch.no_grad():
        kv = (
            mla_attn.kv_b_proj(mla_attn.kv_a_layernorm(compressed_kv))
            .view(
                -1,
                total_len,
                config.num_attention_heads,
                config.qk_nope_head_dim + config.v_head_dim,
            )
            .transpose(1, 2)
        )

    mla_k_nope, mla_val = torch.split(
        kv, [config.qk_nope_head_dim, config.v_head_dim], dim=-1
    )
    mla_key = torch.cat(
        [mla_k_nope, mla_k_pe.expand(-1, config.num_attention_heads, -1, -1)], dim=-1
    )

    assert torch.equal(mla_key, orig_key)
    assert torch.equal(mla_val, orig_val)


@pytest.mark.nightly
def test_kimi_k2_sparse_moe_cpu_parity():
    """
    Verify A2aSparse MLP produces the same results as original DeepseekV3MoE on CPU.
    """

    config_path = os.path.join(os.path.dirname(__file__), "config.json")
    config = DeepseekV3Config.from_json_file(config_path)

    # Scale down for CPU feasibility
    config.n_routed_experts = 16
    config.hidden_size = 256
    config.moe_intermediate_size = 128
    config.num_experts_per_tok = 8
    config.n_shared_experts = 1

    moe = DeepseekV3MoE(config)
    moe = moe.to(torch.bfloat16)
    moe.eval()  # Required for noaux_tc gating

    batch_size = 4
    seq_len = 32
    M = 32  # sparse_matmul token group size
    # Make tokens identical within each M-group: sparse_matmul uses the first
    # token's routing for the entire group.  On CPU (where dispatch doesn't
    # physically group tokens) this ensures routing consistency with the
    # original per-token MoE computation.
    hidden_base = torch.randn(
        batch_size, seq_len // M, 1, config.hidden_size, dtype=torch.bfloat16
    )
    hidden_states = (
        hidden_base.expand(batch_size, seq_len // M, M, -1)
        .reshape(batch_size, seq_len, config.hidden_size)
        .contiguous()
    )

    # Run original BEFORE creating A2aSparse (adapter stacks weights from experts)
    with torch.no_grad():
        original_out = moe(hidden_states)

    a2a_wrapper = enable_sparse_mlp(
        moe,
        mesh=(1, 1),
        config=config,
    )

    with torch.no_grad():
        a2a_out = a2a_wrapper(hidden_states)

    def compute_pcc(x, y):
        x_flat, y_flat = x.flatten().float(), y.flatten().float()
        vx, vy = x_flat - x_flat.mean(), y_flat - y_flat.mean()
        denom = vx.norm() * vy.norm()
        if denom == 0:
            return 1.0 if torch.allclose(x_flat, y_flat) else 0.0
        return float((vx @ vy) / denom)

    pcc = compute_pcc(original_out, a2a_out)
    print(f"PCC (original vs A2aSparse): {pcc:.6f}")
    assert pcc > 0.99, f"Output PCC too low: {pcc:.6f}"
