# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pytest
import torch
import torch_xla
import torch_xla.runtime as xr
from infra import Framework, run_graph_test
from torch_xla.distributed.spmd import Mesh
from transformers import AutoConfig
from transformers.models.gpt_oss.modeling_gpt_oss import GptOssDecoderLayer
from tt_torch.sparse_mlp import enable_sparse_mlp


@pytest.mark.nightly
@pytest.mark.llmbox
def test_gpt_oss_120b_layer():
    xr.set_device_type("TT")
    torch_xla.runtime.use_spmd()

    config = AutoConfig.from_pretrained("openai/gpt-oss-120b", trust_remote_code=True)
    config._attn_implementation = "eager"
    config.num_hidden_layers = 1

    layer = GptOssDecoderLayer(config, layer_idx=0)
    layer = layer.to(torch.bfloat16)
    layer.eval()

    batch_size = 32
    seq_len = 1
    hidden_states = torch.randn(
        (batch_size, seq_len, config.hidden_size), dtype=torch.bfloat16
    )
    attention_mask = torch.rand(batch_size, 1, seq_len, seq_len, dtype=torch.bfloat16)
    position_ids = torch.arange(seq_len).unsqueeze(0)
    cos_sin = torch.rand(
        batch_size, seq_len, config.head_dim // 2, dtype=torch.bfloat16
    )
    position_embeddings = (cos_sin, cos_sin)

    num_devices = xr.global_runtime_device_count()
    mesh_shape = (4, 8)
    device_ids = np.array(range(num_devices))
    mesh = Mesh(device_ids, mesh_shape, ("batch", "model"))

    enable_sparse_mlp(layer, mesh=mesh_shape, cluster_axis=0, config=config)

    def get_shard_spec(layer, args, kwargs):
        shard_specs = {}

        # Activations
        shard_specs[args[0]] = ("batch", None, None)  # hidden_states
        shard_specs[args[6][0]] = ("batch", None, None)  # cos
        shard_specs[args[6][1]] = ("batch", None, None)  # sin
        shard_specs[args[1]] = ("batch", None, None, None)  # attention_mask

        # Attention weights
        shard_specs[layer.self_attn.q_proj.weight] = ("model", "batch")
        shard_specs[layer.self_attn.q_proj.bias] = ("model",)
        shard_specs[layer.self_attn.k_proj.weight] = ("model", "batch")
        shard_specs[layer.self_attn.k_proj.bias] = ("model",)
        shard_specs[layer.self_attn.v_proj.weight] = ("model", "batch")
        shard_specs[layer.self_attn.v_proj.bias] = ("model",)
        shard_specs[layer.self_attn.o_proj.weight] = ("batch", "model")
        shard_specs[layer.self_attn.o_proj.bias] = ("batch",)
        shard_specs[layer.self_attn.sinks] = (None,)

        # MLP / Sparse MoE weights
        shard_specs[layer.mlp.router.weight] = (None, "batch")
        shard_specs[layer.mlp.router.bias] = (None,)
        shard_specs[layer.mlp.experts.gate_up_proj] = (("batch", "model"), None, None)
        shard_specs[layer.mlp.experts.gate_up_proj_bias] = (("batch", "model"), None)
        shard_specs[layer.mlp.experts.down_proj] = (("batch", "model"), None, None)
        shard_specs[layer.mlp.experts.down_proj_bias] = (("batch", "model"), None)

        # LayerNorm
        shard_specs[layer.input_layernorm.weight] = ("batch",)
        shard_specs[layer.post_attention_layernorm.weight] = ("batch",)

        return shard_specs

    run_graph_test(
        layer,
        [
            hidden_states,
            attention_mask,
            position_ids,
            None,  # past_key_values
            False,  # use_cache
            None,  # cache_position
            position_embeddings,
        ],
        framework=Framework.TORCH,
        mesh=mesh,
        shard_spec_fn=get_shard_spec,
    )
