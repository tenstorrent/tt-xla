# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pytest
import torch
import torch_xla
import torch_xla.runtime as xr
from infra import run_graph_test
from infra.evaluators.evaluation_config import ComparisonConfig, PccConfig
from infra.utilities import Framework
from torch_xla.distributed.spmd import Mesh
from transformers import AutoModelForCausalLM

from third_party.tt_forge_models.glm.causal_lm.pytorch.loader import (
    ModelLoader,
    ModelVariant,
)

# Modified load model function to load only one layer
'''
def load_model(self, dtype_override=None, num_layers=None, config=None):
    """Load and return the model instance for this instance's variant.

    Args:
        dtype_override: Optional torch.dtype to override the model's default dtype.
                        If not provided, the model will use its default dtype (typically float32).
        num_layers: Optional number of layers to load. If not provided, all layers are loaded.

    Returns:
        torch.nn.Module: The model instance for causal LM.
    """
    # Get the pretrained model name from the instance's variant config
    pretrained_model_name = self._variant_config.pretrained_model_name

    # Ensure tokenizer is loaded
    if self.tokenizer is None:
        self._load_tokenizer(dtype_override=dtype_override)

    if config is not None:
        model = AutoModelForCausalLM.from_config(config, torch_dtype=dtype_override)
        model.eval()
        self.model = model
        self.config = model.config
        return model

    # Load the model with dtype override if specified
    model_kwargs = {}
    if dtype_override is not None:
        model_kwargs["torch_dtype"] = dtype_override

    model = AutoModelForCausalLM.from_pretrained(
        pretrained_model_name, **model_kwargs
    )
    if num_layers is not None:
        model.model.layers = model.model.layers[:num_layers]

    model.eval()
    self.model = model
    self.config = model.config

    return model
'''


@pytest.mark.parametrize("tp_bool", [True, False])
def test_glm_single_layer(tp_bool):
    """Test GLM single layer with tensor parallel sharding."""
    # Load the model (need to use modified load model function from above)
    loader = ModelLoader(variant=ModelVariant.GLM_4_5)
    config = loader.load_config()
    config.num_hidden_layers = 1
    # model = loader.load_model(dtype_override=torch.bfloat16, config=config)
    # print(f"Model: {model}")
    model = AutoModelForCausalLM.from_config(config=config, torch_dtype=torch.bfloat16)
    print(f"Model: {model}")

    # Extract single layer
    layer = model.model.layers[0]

    print(f"\n{'='*60}")
    print(f"Model and layer loaded, input created started.")
    print(f"{'='*60}\n")

    # Create inputs
    batch_size = 1
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

    # Setup mesh for tensor parallel
    if tp_bool:
        num_devices = xr.global_runtime_device_count()
        mesh_shape, mesh_axes = loader.get_mesh_config(num_devices)
        device_ids = np.array(range(num_devices))
        mesh = Mesh(device_ids, mesh_shape, mesh_axes)

        # Define shard spec function
        def get_shard_spec(layer, args, kwargs):
            """Returns shard specifications for the layer's parameters."""
            shard_specs = {}
            shard_specs[layer.mlp.up_proj.weight] = ("model", "batch")
            shard_specs[layer.mlp.gate_proj.weight] = ("model", "batch")
            shard_specs[layer.mlp.down_proj.weight] = ("batch", "model")

            shard_specs[layer.input_layernorm.weight] = ("batch",)
            shard_specs[layer.post_attention_layernorm.weight] = ("batch",)
            shard_specs[layer.self_attn.q_proj.weight] = ("model", "batch")
            shard_specs[layer.self_attn.q_proj.bias] = ("model",)
            shard_specs[layer.self_attn.k_proj.weight] = ("model", "batch")
            shard_specs[layer.self_attn.k_proj.bias] = ("model",)
            shard_specs[layer.self_attn.v_proj.weight] = ("model", "batch")
            shard_specs[layer.self_attn.v_proj.bias] = ("model",)
            shard_specs[layer.self_attn.o_proj.weight] = ("batch", "model")

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
        None,
        position_embeddings,
    ]

    # Run the graph test with sharding
    comparison_config = ComparisonConfig(pcc=PccConfig(required_pcc=0.99))

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


"""
Model: Glm4MoeForCausalLM(
  (model): Glm4MoeModel(
    (embed_tokens): Embedding(151552, 5120, padding_idx=151329)
    (layers): ModuleList(
      (0-40): Glm4MoeDecoderLayer(
        (self_attn): Glm4MoeAttention(
          (q_proj): Linear(in_features=5120, out_features=12288, bias=True)
          (k_proj): Linear(in_features=5120, out_features=1024, bias=True)
          (v_proj): Linear(in_features=5120, out_features=1024, bias=True)
          (o_proj): Linear(in_features=12288, out_features=5120, bias=False)
          (q_norm): Glm4MoeRMSNorm((128,), eps=1e-05)
          (k_norm): Glm4MoeRMSNorm((128,), eps=1e-05)
        )
        (mlp): Glm4MoeMLP(
          (gate_proj): Linear(in_features=5120, out_features=12288, bias=False)
          (up_proj): Linear(in_features=5120, out_features=12288, bias=False)
          (down_proj): Linear(in_features=12288, out_features=5120, bias=False)
          (act_fn): SiLUActivation()
        )
        (input_layernorm): Glm4MoeRMSNorm((5120,), eps=1e-05)
        (post_attention_layernorm): Glm4MoeRMSNorm((5120,), eps=1e-05)
      )
    )
    (norm): Glm4MoeRMSNorm((5120,), eps=1e-05)
    (rotary_emb): Glm4MoeRotaryEmbedding()
  )
  (lm_head): Linear(in_features=5120, out_features=151552, bias=False)
)"""
