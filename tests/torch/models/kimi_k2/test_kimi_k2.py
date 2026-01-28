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
from torch_xla.distributed.spmd import Mesh

from tests.utils import failed_ttmlir_compilation

from .configuration_deepseek import DeepseekV3Config
from .modeling_deepseek import (
    DeepseekV3Attention,
    DeepseekV3DecoderLayer,
    DeepseekV3ForCausalLM,
)
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

    model = model.to(torch.bfloat16)
    model = model.eval()

    compiled_model = torch.compile(model, backend="tt")

    batch_size = 1
    seq_len = 32
    tokens = torch.randint(0, config.vocab_size, (batch_size, seq_len))

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

    batch_size = 1
    seq_len = 32
    hidden_states = torch.randn(
        (batch_size, seq_len, config.hidden_size), dtype=torch.bfloat16
    )
    attention_mask = torch.rand(batch_size, 1, seq_len, seq_len, dtype=torch.bfloat16)

    device = torch_xla.device()
    attention = attention.to(device)

    num_devices = xr.global_runtime_device_count()
    mesh_shape = (2, 4)
    device_ids = np.array(range(num_devices))
    mesh = Mesh(device_ids, mesh_shape, ("batch", "model"))

    hidden_states = hidden_states.to(device)
    attention_mask = attention_mask.to(device)

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

    run_graph_test(
        attention,
        [hidden_states, attention_mask],
        framework=Framework.TORCH,
        mesh=mesh,
        shard_spec_fn=get_shard_spec,
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

    position_ids = torch.arange(seq_len)
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
        [hidden_states, attention_mask, position_ids, past_key_states],
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

    num_devices = xr.global_runtime_device_count()
    mesh_shape = (2, 4)
    device_ids = np.array(range(num_devices))
    mesh = Mesh(device_ids, mesh_shape, ("_axis_0", "_axis_1"))

    position_ids = torch.arange(seq_len)
    static_cache = MLACache(
        config=config,
        max_batch_size=batch_size,
        max_cache_len=max_cache_len,
        device="cpu",
        dtype=torch.bfloat16,
    )
    past_key_states = static_cache

    device = torch_xla.device()
    layer = layer.to(device)

    num_devices = xr.global_runtime_device_count()
    device_ids = np.array(range(num_devices))

    hidden_states = hidden_states.to(device)
    attention_mask = attention_mask.to(device)

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
        [hidden_states, attention_mask, position_ids, past_key_states],
        framework=Framework.TORCH,
        mesh=mesh,
        shard_spec_fn=get_shard_spec,
    )
