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
from tt_torch.sparse_mlp import create_a2a_from_deepseek_v3_moe

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
    # Additive causal mask: 0 for attend, -inf for mask out
    # TTIR requires 4D mask: [batch, heads, seq_len, seq_len]
    attention_mask = torch.zeros(
        batch_size, 1, seq_len, max_cache_len, dtype=torch.bfloat16
    )
    attention_mask.masked_fill_(
        ~torch.ones(seq_len, max_cache_len).bool().tril(), float("-inf")
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
    # Additive causal mask: 0 for attend, -inf for mask out
    # TTIR requires 4D mask: [batch, heads, seq_len, seq_len]
    attention_mask = torch.zeros(
        batch_size, 1, seq_len, max_cache_len, dtype=torch.bfloat16
    )
    attention_mask.masked_fill_(
        ~torch.ones(seq_len, max_cache_len).bool().tril(), float("-inf")
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
    for layer in static_cache.layers:
        layer.lazy_initialization(
            torch.zeros(batch_size, 1, 1, config.kv_lora_rank, dtype=torch.bfloat16),
            torch.zeros(
                batch_size, 1, 1, config.qk_rope_head_dim, dtype=torch.bfloat16
            ),
        )
    past_key_states = static_cache

    def get_shard_spec(attention, args, kwargs):
        shard_specs = {}

        shard_specs[args[0]] = ("_axis_1", None, "_axis_0")
        shard_specs[args[1]] = ("_axis_1", None, None, None)
        shard_specs[args[3].layers[0].compressed_kv] = ("_axis_1", None, None, None)
        shard_specs[args[3].layers[0].k_pe] = ("_axis_1", None, None, None)

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
    # Additive causal mask: 0 for attend, -inf for mask out
    # TTIR requires 4D mask: [batch, heads, seq_len, seq_len]
    attention_mask = torch.zeros(
        batch_size, 1, seq_len, max_cache_len, dtype=torch.bfloat16
    )
    attention_mask.masked_fill_(
        ~torch.ones(seq_len, max_cache_len).bool().tril(), float("-inf")
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
    for cache_layer in static_cache.layers:
        cache_layer.lazy_initialization(
            torch.zeros(batch_size, 1, 1, config.kv_lora_rank, dtype=torch.bfloat16),
            torch.zeros(
                batch_size, 1, 1, config.qk_rope_head_dim, dtype=torch.bfloat16
            ),
        )
    past_key_states = static_cache

    num_devices = xr.global_runtime_device_count()
    device_ids = np.array(range(num_devices))

    def get_shard_spec(layer, args, kwargs):
        shard_specs = {}

        shard_specs[args[0]] = ("_axis_1", None, "_axis_0")
        shard_specs[args[1]] = ("_axis_1", None, None, None)
        shard_specs[args[3].layers[0].compressed_kv] = ("_axis_1", None, None, None)
        shard_specs[args[3].layers[0].k_pe] = ("_axis_1", None, None, None)

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
@pytest.mark.parametrize("batch_size", [32])
@pytest.mark.parametrize("seq_len", [1, 32])
def test_kimi_k2_layer_sparse_moe(batch_size, seq_len):
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

        shard_specs[args[0]] = ("_axis_1", None, "_axis_0")
        shard_specs[args[1]] = ("_axis_1", None, None, None)
        shard_specs[args[3].layers[1].compressed_kv] = ("_axis_1", None, None, None)
        shard_specs[args[3].layers[1].k_pe] = ("_axis_1", None, None, None)

        shard_specs[layer.self_attn.q_b_proj.weight] = ("_axis_0", None)
        shard_specs[layer.self_attn.kv_b_proj.weight] = ("_axis_0", None)
        shard_specs[layer.self_attn.o_proj.weight] = (None, "_axis_0")

        shard_specs[layer.self_attn.q_a_proj.weight] = (None, "_axis_0")
        shard_specs[layer.self_attn.kv_a_proj_with_mqa.weight] = (None, "_axis_0")

        # A2aSparseMLP: experts compound-sharded (axis_0, axis_1)
        mlp_wrapper = layer.mlp
        mlp = mlp_wrapper.mlp if hasattr(mlp_wrapper, "mlp") else mlp_wrapper
        shard_specs[mlp.router.gate.weight] = (None, "_axis_0")
        shard_specs[mlp.experts.gate_proj] = (
            ("_axis_0", "_axis_1"),
            None,
            None,
        )
        shard_specs[mlp.experts.up_proj] = (
            ("_axis_0", "_axis_1"),
            None,
            None,
        )
        shard_specs[mlp.experts.down_proj] = (
            ("_axis_0", "_axis_1"),
            None,
            None,
        )

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

    comparison_config = ComparisonConfig(
        pcc=PccConfig(enabled=True, required_pcc=0.98),
    )

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
def test_kimi_k2_causal_lm_mla_cache_prefill_decode():
    """
    CPU test validating prefill + decode of DeepseekV3ForCausalLM with MLACache.

    Uses a 2-layer model (layer 0 = dense MLP, layer 1 = MoE) with a small
    config to keep the test fast. Verifies that:
      - Prefill fills the cache and returns logits of the right shape.
      - Multiple decode steps advance cache_position and return single-token logits.
      - The cache is actually written to (non-zero entries after prefill).
    """
    config = DeepseekV3Config(
        vocab_size=100,
        hidden_size=64,
        intermediate_size=128,
        moe_intermediate_size=32,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=4,
        q_lora_rank=32,
        kv_lora_rank=16,
        qk_rope_head_dim=16,
        v_head_dim=8,
        qk_nope_head_dim=8,
        n_shared_experts=1,
        n_routed_experts=4,        # 4 experts total
        num_experts_per_tok=2,     # top-2 selected per token
        first_k_dense_replace=1,   # layer 0 dense, layer 1 MoE
        n_group=2,                 # 2 groups of 2 experts each (noaux_tc requires >=2 per group)
        topk_group=1,              # pick from 1 group
        max_position_embeddings=64,
        rope_scaling=None,
        attention_bias=False,
        attention_dropout=0.0,
    )

    BATCH_SIZE = 2
    PREFILL_LEN = 8
    DECODE_STEPS = 3
    MAX_CACHE_LEN = PREFILL_LEN + DECODE_STEPS

    model = DeepseekV3ForCausalLM(config).eval()

    mla_cache = MLACache(config, max_cache_len=MAX_CACHE_LEN)

    # --- Prefill ---
    prefill_tokens = torch.randint(0, config.vocab_size, (BATCH_SIZE, PREFILL_LEN))
    prefill_cache_position = torch.arange(PREFILL_LEN)

    with torch.no_grad():
        prefill_output = model(
            input_ids=prefill_tokens,
            past_key_values=mla_cache,
            cache_position=prefill_cache_position,
            use_cache=True,
        )

    assert prefill_output.logits.shape == (BATCH_SIZE, PREFILL_LEN, config.vocab_size)

    # Cache must be initialized and written to after prefill
    for layer_cache in mla_cache.layers:
        assert layer_cache.is_initialized
        assert layer_cache.compressed_kv[:, 0, :PREFILL_LEN, :].abs().sum() > 0

    # --- Decode ---
    next_token = prefill_output.logits[:, -1].argmax(dim=-1, keepdim=True)

    for step in range(DECODE_STEPS):
        decode_cache_position = torch.tensor([PREFILL_LEN + step])

        with torch.no_grad():
            decode_output = model(
                input_ids=next_token,
                past_key_values=mla_cache,
                cache_position=decode_cache_position,
                use_cache=True,
            )

        assert decode_output.logits.shape == (BATCH_SIZE, 1, config.vocab_size)
        next_token = decode_output.logits[:, -1].argmax(dim=-1, keepdim=True)


@pytest.mark.nightly
@pytest.mark.galaxy
def test_kimi_k2_moe_compile():
    """
    Torch graph test for DeepseekV3MoE via SparseMLP compiled and run on Galaxy (4x8 mesh).

    Uses a small config (hidden_size=64, 4 routed experts) to keep instantiation fast.
    The raw DeepseekV3MoE cannot be compiled because moe_infer uses
    tokens_per_expert.cpu().tolist() for data-dependent loop control, which breaks
    dynamo's fake-tensor tracing. This test converts the MoE to A2aSparseMLPWithSharedExperts
    via create_a2a_from_deepseek_v3_moe, which replaces the per-expert Python loop with
    torch.ops.tt.sparse_matmul and static tensor routing.
    """
    xr.set_device_type("TT")
    torch_xla.runtime.use_spmd()
    os.environ["CONVERT_SHLO_TO_SHARDY"] = "1"

    config = DeepseekV3Config(
        hidden_size=64,
        intermediate_size=128,
        moe_intermediate_size=32,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=4,
        q_lora_rank=32,
        kv_lora_rank=16,
        qk_rope_head_dim=16,
        v_head_dim=8,
        qk_nope_head_dim=8,
        n_shared_experts=1,
        n_routed_experts=32,
        num_experts_per_tok=2,
        first_k_dense_replace=1,
        n_group=4,
        topk_group=1,
        max_position_embeddings=64,
        rope_scaling=None,
        attention_bias=False,
        attention_dropout=0.0,
    )

    moe = DeepseekV3MoE(config).to(torch.bfloat16).eval()

    BATCH_SIZE = 32
    SEQ_LEN = 16

    num_devices = xr.global_runtime_device_count()
    mesh_shape = (4, 8)
    device_ids = np.array(range(num_devices))
    mesh = Mesh(device_ids, mesh_shape, ("batch", "model"))

    # Convert to A2aSparseMLP — replaces the data-dependent moe_infer loop with
    # sparse_matmul. cluster_axis=0 dispatches along the batch axis (4 devices),
    # dispatch_devices=mesh_shape[0]=4.
    sparse_moe = create_a2a_from_deepseek_v3_moe(
        moe_module=moe,
        config=config,
        num_devices=num_devices,
        cluster_axis=0,
        dispatch_devices=mesh_shape[0],
    ).eval()

    hidden_states = torch.randn(
        BATCH_SIZE, SEQ_LEN, config.hidden_size, dtype=torch.bfloat16
    )

    def get_shard_spec(sparse_moe, args, kwargs):
        shard_specs = {}
        shard_specs[args[0]] = ("batch", None, None)
        # Gate (router) weight — replicated across model axis
        shard_specs[sparse_moe.mlp.router.gate.weight] = (None, "batch")
        # Stacked expert weights — compound-sharded on expert dim across both axes
        shard_specs[sparse_moe.mlp.experts.gate_up_proj] = (("batch", "model"), None, None)
        shard_specs[sparse_moe.mlp.experts.gate_up_proj_bias] = (("batch", "model"), None)
        shard_specs[sparse_moe.mlp.experts.down_proj] = (("batch", "model"), None, None)
        shard_specs[sparse_moe.mlp.experts.down_proj_bias] = (("batch", "model"), None)
        # Shared experts — column/row parallel
        if sparse_moe.shared_experts is not None:
            shard_specs[sparse_moe.shared_experts.gate_proj.weight] = ("model", "batch")
            shard_specs[sparse_moe.shared_experts.up_proj.weight] = ("model", "batch")
            shard_specs[sparse_moe.shared_experts.down_proj.weight] = ("batch", "model")
        return shard_specs

    run_graph_test(
        sparse_moe,
        [hidden_states],
        framework=Framework.TORCH,
        mesh=mesh,
        shard_spec_fn=get_shard_spec,
    )


def _kimi_k2_shard_model(model, mesh):
    """Apply tensor-parallel sharding to a Kimi K2 model using the given mesh.

    Mesh axis names must be ("batch", "model").
    """
    xs.mark_sharding(model.model.embed_tokens.weight, mesh, (None, "batch"))
    xs.mark_sharding(model.model.norm.weight, mesh, ("batch",))
    xs.mark_sharding(model.lm_head.weight, mesh, ("model", "batch"))

    for layer in model.model.layers:
        # MLA attention
        xs.mark_sharding(layer.self_attn.q_a_proj.weight, mesh, (None, "batch"))
        xs.mark_sharding(layer.self_attn.q_b_proj.weight, mesh, ("model", None))
        xs.mark_sharding(layer.self_attn.q_a_layernorm.weight, mesh, (None,))
        xs.mark_sharding(layer.self_attn.kv_a_proj_with_mqa.weight, mesh, (None, "batch"))
        xs.mark_sharding(layer.self_attn.kv_b_proj.weight, mesh, ("model", None))
        xs.mark_sharding(layer.self_attn.kv_a_layernorm.weight, mesh, (None,))
        xs.mark_sharding(layer.self_attn.o_proj.weight, mesh, ("batch", "model"))

        if isinstance(layer.mlp, DeepseekV3MoE):
            xs.mark_sharding(layer.mlp.gate.weight, mesh, (None, "batch"))
            for expert in layer.mlp.experts:
                xs.mark_sharding(expert.gate_proj.weight, mesh, ("model", "batch"))
                xs.mark_sharding(expert.up_proj.weight, mesh, ("model", "batch"))
                xs.mark_sharding(expert.down_proj.weight, mesh, ("batch", "model"))
            if getattr(layer.mlp, "shared_experts", None) is not None:
                xs.mark_sharding(layer.mlp.shared_experts.gate_proj.weight, mesh, ("model", "batch"))
                xs.mark_sharding(layer.mlp.shared_experts.up_proj.weight, mesh, ("model", "batch"))
                xs.mark_sharding(layer.mlp.shared_experts.down_proj.weight, mesh, ("batch", "model"))
        else:
            xs.mark_sharding(layer.mlp.gate_proj.weight, mesh, ("model", "batch"))
            xs.mark_sharding(layer.mlp.up_proj.weight, mesh, ("model", "batch"))
            xs.mark_sharding(layer.mlp.down_proj.weight, mesh, ("batch", "model"))

        xs.mark_sharding(layer.input_layernorm.weight, mesh, ("batch",))
        xs.mark_sharding(layer.post_attention_layernorm.weight, mesh, ("batch",))


@pytest.mark.nightly
@pytest.mark.galaxy
def test_kimi_k2_causal_lm_mla_cache_compiled_galaxy():
    """
    Compiled prefill + decode for a 2-layer Kimi K2 model with MLACache on Galaxy (4x8 mesh).

    Uses the real Kimi K2 config (hidden_size=7168, 384 experts, …) with
    num_hidden_layers=2 so we can see which XLA graphs fail to compile.
    Runs prefill (batch=32, seq=16) then three decode steps.
    """
    xr.set_device_type("TT")
    xr.use_spmd()
    os.environ["CONVERT_SHLO_TO_SHARDY"] = "1"

    config_path = os.path.join(os.path.dirname(__file__), "config.json")
    config = DeepseekV3Config.from_json_file(config_path)
    config.num_hidden_layers = 2

    BATCH_SIZE = 32
    PREFILL_LEN = 16
    DECODE_STEPS = 3
    MAX_CACHE_LEN = PREFILL_LEN + DECODE_STEPS

    # Galaxy mesh (4 rows × 8 columns)
    num_devices = xr.global_runtime_device_count()
    mesh_shape = (4, 8)
    device_ids = np.array(range(num_devices))
    mesh = Mesh(device_ids, mesh_shape, ("batch", "model"))

    device = torch_xla.device()

    # Build model in bfloat16
    model = DeepseekV3ForCausalLM(config).to(torch.bfloat16).eval()
    model = model.to(device)
    _kimi_k2_shard_model(model, mesh)

    # Pre-initialise MLA cache on CPU then transfer to device
    mla_cache = MLACache(config, max_cache_len=MAX_CACHE_LEN)
    dummy_kv = torch.zeros((BATCH_SIZE, 1, 1, config.kv_lora_rank), dtype=torch.bfloat16)
    dummy_pe = torch.zeros((BATCH_SIZE, 1, 1, config.qk_rope_head_dim), dtype=torch.bfloat16)
    for cache_layer in mla_cache.layers:
        cache_layer.lazy_initialization(dummy_kv, dummy_pe)

    for cache_layer in mla_cache.layers:
        cache_layer.compressed_kv = cache_layer.compressed_kv.to(device)
        cache_layer.k_pe = cache_layer.k_pe.to(device)
        cache_layer.keys = cache_layer.compressed_kv
        cache_layer.values = cache_layer.k_pe
        torch._dynamo.mark_static_address(cache_layer.compressed_kv)
        torch._dynamo.mark_static_address(cache_layer.k_pe)
        # compressed_kv / k_pe are in the shared latent space (not head-sharded);
        # replicate across the model axis and shard across the batch axis.
        xs.mark_sharding(cache_layer.compressed_kv, mesh, (None, None, None, None))
        xs.mark_sharding(cache_layer.k_pe, mesh, (None, None, None, None))

    compiled_model = torch.compile(model, backend="tt")

    # --- Prefill ---
    prefill_tokens = torch.randint(0, config.vocab_size, (BATCH_SIZE, PREFILL_LEN)).to(device)
    prefill_cache_position = torch.arange(PREFILL_LEN).to(device)

    with torch.no_grad():
        prefill_output = compiled_model(
            input_ids=prefill_tokens,
            past_key_values=mla_cache,
            cache_position=prefill_cache_position,
            use_cache=True,
        )

    prefill_logits = prefill_output.logits.to("cpu")
    assert prefill_logits.shape == (BATCH_SIZE, PREFILL_LEN, config.vocab_size)

    # --- Decode ---
    next_token = prefill_logits[:, -1].argmax(dim=-1, keepdim=True).to(device)

    for step in range(DECODE_STEPS):
        decode_cache_position = torch.tensor([PREFILL_LEN + step]).to(device)

        with torch.no_grad():
            decode_output = compiled_model(
                input_ids=next_token,
                past_key_values=mla_cache,
                cache_position=decode_cache_position,
                use_cache=True,
            )

        decode_logits = decode_output.logits.to("cpu")
        assert decode_logits.shape == (BATCH_SIZE, 1, config.vocab_size)
        next_token = decode_logits[:, -1].argmax(dim=-1, keepdim=True).to(device)
