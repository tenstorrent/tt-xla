# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pytest
import torch
import torch_xla
import torch_xla.runtime as xr
from infra import Framework, run_graph_test
from infra.evaluators import ComparisonConfig, PccConfig
from torch_xla.distributed.spmd import Mesh
from transformers.models.glm4_moe.configuration_glm4_moe import Glm4MoeConfig
from transformers.models.glm4_moe.modeling_glm4_moe import (
    Glm4MoeDecoderLayer,
    Glm4MoeModel,
)
from tt_torch.sparse_mlp import enable_sparse_mlp


@pytest.mark.nightly
@pytest.mark.llmbox
def test_glm4_moe_layer_sparse_moe():
    """Test single MoE decoder layer with A2aSparseMLP on (2,4) mesh."""
    xr.set_device_type("TT")
    torch_xla.runtime.use_spmd()

    config = Glm4MoeConfig.from_pretrained("zai-org/GLM-4.7")
    # first_k_dense_replace=3, so need at least 4 layers for MoE
    config.num_hidden_layers = 4
    config.use_cache = False
    config._attn_implementation = "eager"

    # layer_idx=3 >= first_k_dense_replace=3 -> MoE layer
    layer = Glm4MoeDecoderLayer(config, layer_idx=3)
    layer = layer.eval().to(torch.bfloat16)

    batch_size = 64
    seq_len = 1

    mesh_shape = (2, 4)
    enable_sparse_mlp(layer, mesh=mesh_shape, cluster_axis=0, config=config)

    hidden_states = torch.randn(
        (batch_size, seq_len, config.hidden_size), dtype=torch.bfloat16
    )
    position_ids = torch.arange(seq_len).unsqueeze(0)
    # position_embeddings: (cos, sin) each [batch, seq, head_dim]
    head_dim = config.hidden_size // config.num_attention_heads
    rotary_dim = int(head_dim * config.partial_rotary_factor)
    cos = torch.randn(batch_size, seq_len, rotary_dim, dtype=torch.bfloat16)
    sin = torch.randn(batch_size, seq_len, rotary_dim, dtype=torch.bfloat16)

    num_devices = xr.global_runtime_device_count()
    device_ids = np.array(range(num_devices))
    mesh = Mesh(device_ids, mesh_shape, ("_axis_0", "_axis_1"))

    def get_shard_spec(layer, args, kwargs):
        shard_specs = {}

        # hidden_states: [batch, seq, hidden]
        shard_specs[args[0]] = ("_axis_1", None, "_axis_0")

        # Attention weights
        attn = layer.self_attn
        shard_specs[attn.q_proj.weight] = ("_axis_0", None)
        shard_specs[attn.k_proj.weight] = ("_axis_0", None)
        shard_specs[attn.v_proj.weight] = ("_axis_0", None)
        shard_specs[attn.o_proj.weight] = (None, "_axis_0")
        if attn.q_proj.bias is not None:
            shard_specs[attn.q_proj.bias] = ("_axis_0",)
            shard_specs[attn.k_proj.bias] = ("_axis_0",)
            shard_specs[attn.v_proj.bias] = ("_axis_0",)

        # MoE (A2aSparseMLPWithSharedExperts)
        mlp_wrapper = layer.mlp
        mlp = mlp_wrapper.mlp if hasattr(mlp_wrapper, "mlp") else mlp_wrapper
        shard_specs[mlp.router.gate.weight] = (None, "_axis_0")
        # shard_specs[mlp.experts.gate_proj] = (
        #     ("_axis_0", "_axis_1"), None, None,
        # )
        # shard_specs[mlp.experts.up_proj] = (
        #     ("_axis_0", "_axis_1"), None, None,
        # )
        shard_specs[mlp.experts.gate_up_proj] = (
            ("_axis_0", "_axis_1"), None, None,
        )
       
        shard_specs[mlp.experts.down_proj] = (
            ("_axis_0", "_axis_1"), None, None,
        )
        shard_specs[mlp.experts.gate_up_proj_bias] = (("_axis_0", "_axis_1"), None)
        # shard_specs[mlp.experts.gate_proj_bias] = (("_axis_0", "_axis_1"), None)
        # shard_specs[mlp.experts.up_proj_bias] = (("_axis_0", "_axis_1"), None)
        shard_specs[mlp.experts.down_proj_bias] = (("_axis_0", "_axis_1"), None)

        # # Shared experts
        # shared = getattr(mlp_wrapper, "shared_experts", None)
        # if shared is not None:
        #     shard_specs[shared.gate_proj.weight] = (None, "_axis_0")
        #     shard_specs[shared.up_proj.weight] = (None, "_axis_0")
        #     shard_specs[shared.down_proj.weight] = ("_axis_0", None)

        # Norms
        shard_specs[layer.input_layernorm.weight] = ("_axis_0",)
        shard_specs[layer.post_attention_layernorm.weight] = ("_axis_0",)

        return shard_specs

    comparison_config = ComparisonConfig(
        pcc=PccConfig(enabled=True, required_pcc=0.95),
    )

    run_graph_test(
        layer,
        [hidden_states, None, position_ids, None, False, None, (cos, sin)],
        framework=Framework.TORCH,
        mesh=mesh,
        shard_spec_fn=get_shard_spec,
        comparison_config=comparison_config,
    )


@pytest.mark.nightly
@pytest.mark.llmbox
def test_glm4_moe_full_sparse_moe():
    """Test full Glm4MoeModel with A2aSparseMLP on (2,4) mesh."""
    xr.set_device_type("TT")
    torch_xla.runtime.use_spmd()

    config = Glm4MoeConfig.from_pretrained("zai-org/GLM-4.7")
    # first_k_dense_replace=3, so need at least 4 layers for MoE
    config.num_hidden_layers = 4
    config.use_cache = False
    config._attn_implementation = "eager"

    model = Glm4MoeModel(config)
    model = model.to(torch.bfloat16)

    mesh_shape = (2, 4)
    enable_sparse_mlp(model, mesh=mesh_shape, cluster_axis=0, config=config)
    model.eval()

    batch_size = 64
    seq_len = 1
    input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len))
    position_ids = torch.arange(seq_len).unsqueeze(0)

    num_devices = xr.global_runtime_device_count()
    device_ids = np.array(range(num_devices))
    mesh = Mesh(device_ids, mesh_shape, ("_axis_0", "_axis_1"))

    def get_shard_spec(model, args, kwargs):
        shard_specs = {}

        # input_ids: [batch, seq]
        shard_specs[args[0]] = ("_axis_1", None)

        # Embedding
        shard_specs[model.embed_tokens.weight] = (None, "_axis_0")

        for decoder_layer in model.layers:
            # Attention
            attn = decoder_layer.self_attn
            shard_specs[attn.q_proj.weight] = ("_axis_0", None)
            shard_specs[attn.k_proj.weight] = ("_axis_0", None)
            shard_specs[attn.v_proj.weight] = ("_axis_0", None)
            shard_specs[attn.o_proj.weight] = (None, "_axis_0")
            if attn.q_proj.bias is not None:
                shard_specs[attn.q_proj.bias] = ("_axis_0",)
                shard_specs[attn.k_proj.bias] = ("_axis_0",)
                shard_specs[attn.v_proj.bias] = ("_axis_0",)

            # MLP (MoE or dense)
            mlp_wrapper = decoder_layer.mlp
            if hasattr(mlp_wrapper, "mlp"):
                # A2aSparseMLPWithSharedExperts
                mlp = mlp_wrapper.mlp
                shard_specs[mlp.router.gate.weight] = (None, "_axis_0")
                shard_specs[mlp.experts.gate_proj] = (
                    ("_axis_0", "_axis_1"), None, None,
                )
                shard_specs[mlp.experts.up_proj] = (
                    ("_axis_0", "_axis_1"), None, None,
                )
                shard_specs[mlp.experts.down_proj] = (
                    ("_axis_0", "_axis_1"), None, None,
                )
                shard_specs[mlp.experts.gate_proj_bias] = (("_axis_0", "_axis_1"), None)
                shard_specs[mlp.experts.up_proj_bias] = (("_axis_0", "_axis_1"), None)
                shard_specs[mlp.experts.down_proj_bias] = (("_axis_0", "_axis_1"), None)

                # Shared experts
                shared = getattr(mlp_wrapper, "shared_experts", None)
                if shared is not None:
                    shard_specs[shared.gate_proj.weight] = (None, "_axis_0")
                    shard_specs[shared.up_proj.weight] = (None, "_axis_0")
                    shard_specs[shared.down_proj.weight] = ("_axis_0", None)
            else:
                # Dense MLP
                shard_specs[mlp_wrapper.gate_proj.weight] = (
                    "_axis_1",
                    "_axis_0",
                )
                shard_specs[mlp_wrapper.up_proj.weight] = (
                    "_axis_1",
                    "_axis_0",
                )
                shard_specs[mlp_wrapper.down_proj.weight] = (
                    "_axis_0",
                    "_axis_1",
                )

            # Norms
            shard_specs[decoder_layer.input_layernorm.weight] = ("_axis_0",)
            shard_specs[decoder_layer.post_attention_layernorm.weight] = (
                "_axis_0",
            )

        # Final norm
        shard_specs[model.norm.weight] = ("_axis_0",)

        return shard_specs

    comparison_config = ComparisonConfig(
        pcc=PccConfig(enabled=True, required_pcc=0.95),
    )

    run_graph_test(
        model,
        [input_ids],
        framework=Framework.TORCH,
        mesh=mesh,
        shard_spec_fn=get_shard_spec,
        comparison_config=comparison_config,
    )
