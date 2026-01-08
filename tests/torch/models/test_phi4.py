# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pytest
import torch
import torch_xla
import torch_xla.runtime as xr
from infra import run_graph_test
from infra.comparators.comparison_config import ComparisonConfig, PccConfig
from infra.utilities import Framework
from torch_xla.distributed.spmd import Mesh

from third_party.tt_forge_models.phi4.causal_lm.pytorch.loader import ModelLoader


@pytest.mark.parametrize("tp_bool", [True, False])
def test_phi4_1layer(tp_bool):
    """Test Phi4 single layer with tensor parallel sharding."""
    # Load the model (full model to get config)
    loader = ModelLoader()
    model = loader.load_model(dtype_override=torch.bfloat16)

    # Extract single layer
    layer = model.model.layers[0]

    print(f"\n{'='*60}")
    print(f"Model and layer loaded, input created started.")
    print(f"{'='*60}\n")

    # Create inputs
    batch_size = 4
    seq_len = 128
    head_dim = model.config.hidden_size // model.config.num_attention_heads
    hidden_states = torch.randn(
        batch_size, seq_len, model.config.hidden_size, dtype=torch.bfloat16
    )
    attention_mask = torch.ones(batch_size, 1, seq_len, seq_len, dtype=torch.bfloat16)
    cos_sin = torch.randn(batch_size, seq_len, head_dim, dtype=torch.bfloat16)
    position_embeddings = (cos_sin, cos_sin)

    # Set device type
    xr.set_device_type("TT")

    # Setup mesh for tensor parallel (only if running on llmbox)
    if tp_bool:
        num_devices = xr.global_runtime_device_count()
        mesh_shape = (batch_size, num_devices // batch_size)
        device_ids = np.array(range(num_devices))
        mesh = Mesh(device_ids, mesh_shape, ("batch", "model"))

        # Define shard spec function
        def get_shard_spec(layer_model, args, kwargs):
            """Returns shard specifications for the layer's parameters."""
            shard_specs = {}

            # Shard attention weights
            shard_specs[layer_model.self_attn.qkv_proj.weight] = ("model", "batch")
            shard_specs[layer_model.self_attn.o_proj.weight] = ("batch", "model")

            # Shard MLP weights
            shard_specs[layer_model.mlp.gate_up_proj.weight] = ("model", "batch")
            shard_specs[layer_model.mlp.down_proj.weight] = ("batch", "model")

            print(f"Shard specs: {shard_specs}")
            return shard_specs

    else:
        mesh = None
        get_shard_spec = None

    # Prepare inputs as positional args (matching decoder layer signature)
    inputs = [
        hidden_states,
        attention_mask,
        None,
        None,
        False,
        False,
        None,
        position_embeddings,
    ]

    # Run the graph test with sharding
    comparison_config = ComparisonConfig(pcc=PccConfig(required_pcc=0.98))

    print(f"\n{'='*60}")
    print(f"Run graph test")
    print(f"{'='*60}\n")

    run_graph_test(
        layer,
        inputs,
        framework=Framework.TORCH,
        comparison_config=comparison_config,
        mesh=mesh,
        shard_spec_fn=get_shard_spec,
    )

    if tp_bool:
        print(f"\nTest passed with tensor parallel sharding!\n")
    else:
        print(f"\nTest passed without sharding\n")


# Add to loader.py later if things work out maybe??
def get_mesh_config(self, num_devices: int):
    if num_devices == 8:
        mesh_shape = (4, num_devices // 4)
    elif num_devices == 2:
        mesh_shape = (1, num_devices)
    else:
        raise ValueError(
            f"Cannot evenly distribute {self.config.num_attention_heads} heads across {num_devices} devices"
        )
    return mesh_shape, ("batch", "model")


def load_shard_spec(self, model):
    shard_specs = {}
    
    # for layer in model.model.layers:
    for layer in model.model.layers:   
        shard_specs[layer.self_attn.qkv_proj.weight] = ("model", "batch")

        # rowwise_rep = shard second dimension = (None, "model")
        shard_specs[layer.self_attn.o_proj.weight] = ("batch", "model")

        shard_specs[layer.mlp.gate_up_proj.weight] = ("model", "batch")
        shard_specs[layer.mlp.down_proj.weight] = ("batch", "model")

    shard_specs[model.lm_head.weight] = ("model", "batch")
    return shard_specs
