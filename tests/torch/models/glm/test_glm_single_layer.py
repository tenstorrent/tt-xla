# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pytest
import torch
import torch_xla.runtime as xr
from infra import run_graph_test
from infra.evaluators.evaluation_config import ComparisonConfig, PccConfig
from infra.utilities import Framework
from torch_xla.distributed.spmd import Mesh
from transformers import AutoModelForCausalLM

from tests.utils import parametrize_arch
from third_party.tt_forge_models.glm.causal_lm.pytorch.loader import ModelLoader

available_variants = ModelLoader.query_available_variants()


@pytest.mark.nightly
@parametrize_arch(["llmbox"])
@pytest.mark.parametrize("seq_len", [1024])
@pytest.mark.parametrize("variant", available_variants.keys())
def test_glm_single_layer(seq_len, variant, arch):
    """Test GLM single layer with tensor parallel sharding."""

    # Load model with single layer
    loader = ModelLoader(variant=variant)
    config = loader.load_config()
    config.num_hidden_layers = 1
    model = AutoModelForCausalLM.from_config(config=config, torch_dtype=torch.bfloat16)

    # Extract layer from model with single layer
    layer = model.model.layers[0]

    # Create inputs
    batch_size = 2
    head_dim = model.config.head_dim
    hidden_states = torch.randn(
        batch_size, seq_len, model.config.hidden_size, dtype=torch.bfloat16
    )
    attention_mask = torch.ones(batch_size, 1, seq_len, seq_len, dtype=torch.bfloat16)
    cos_sin = torch.randn(batch_size, seq_len, head_dim, dtype=torch.bfloat16)
    position_embeddings = (cos_sin, cos_sin)

    # Set device type
    xr.set_device_type("TT")

    # Setup for tensor parallel
    if arch == "llmbox":
        num_devices = xr.global_runtime_device_count()
        mesh_shape = (batch_size, num_devices // batch_size)
        device_ids = np.array(range(num_devices))
        mesh = Mesh(device_ids, mesh_shape, ("batch", "model"))

        def get_shard_spec(layer, args, kwargs):
            """Returns shard specifications for the layer's parameters."""
            shard_specs = {}

            shard_specs[layer.input_layernorm.weight] = ("batch",)
            shard_specs[layer.post_attention_layernorm.weight] = ("batch",)
            shard_specs[layer.self_attn.q_proj.weight] = ("model", "batch")
            shard_specs[layer.self_attn.q_proj.bias] = ("model",)
            shard_specs[layer.self_attn.k_proj.weight] = ("model", "batch")
            shard_specs[layer.self_attn.k_proj.bias] = ("model",)
            shard_specs[layer.self_attn.v_proj.weight] = ("model", "batch")
            shard_specs[layer.self_attn.v_proj.bias] = ("model",)
            shard_specs[layer.self_attn.o_proj.weight] = ("batch", "model")

            shard_specs[layer.mlp.up_proj.weight] = ("model", "batch")
            shard_specs[layer.mlp.gate_proj.weight] = ("model", "batch")
            shard_specs[layer.mlp.down_proj.weight] = ("batch", "model")

            # input sharding
            shard_specs[args[0]] = ("batch", None, None)
            shard_specs[args[1]] = ("batch", None, None, None)
            shard_specs[args[6][0]] = ("batch", None, None)
            shard_specs[args[6][1]] = ("batch", None, None)

            return shard_specs

    else:
        mesh = None
        get_shard_spec = None

    # Prep inputs (matching decoder layer signature)
    inputs = [
        hidden_states,
        attention_mask,
        None,
        None,
        False,
        None,
        position_embeddings,
    ]

    # Run the graph test
    comparison_config = ComparisonConfig(pcc=PccConfig(required_pcc=0.99))

    run_graph_test(
        layer,
        inputs,
        framework=Framework.TORCH,
        comparison_config=comparison_config,
        mesh=mesh,
        shard_spec_fn=get_shard_spec,
    )
