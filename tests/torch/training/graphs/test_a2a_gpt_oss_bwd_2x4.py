# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
import numpy as np
import pytest
import torch
import torch_xla
import torch_xla.distributed.spmd as xs
import torch_xla.runtime as xr
from benchmark.utils import compute_pcc
from infra.utilities.torch_multichip_utils import enable_spmd
from torch_xla.distributed.spmd import Mesh
from tt_torch.sparse_mlp import enable_sparse_mlp
from transformers import AutoConfig
from transformers.models.gpt_oss.modeling_gpt_oss import GptOssDecoderLayer

from third_party.tt_forge_models.gpt_oss.pytorch.overrides import (
    override_gpt_oss_modules,
)

MESH_SHAPE = (2, 4)
MESH_NAMES = ("_axis_0", "_axis_1")


def _setup_mesh():
    enable_spmd()
    xr.set_device_type("TT")
    num_devices = xr.global_runtime_device_count()
    assert num_devices == 8
    mesh = Mesh(np.arange(num_devices), MESH_SHAPE, MESH_NAMES)
    return mesh, torch_xla.device(), num_devices


def _build_layer(num_experts: int):
    config = AutoConfig.from_pretrained("openai/gpt-oss-20b", trust_remote_code=True)
    config.num_hidden_layers = 1
    config.num_local_experts = num_experts
    config._attn_implementation = "eager"
    layer = GptOssDecoderLayer(config, layer_idx=0).to(torch.float32)
    override_gpt_oss_modules(layer)
    return layer, config


def _shard_decoder_layer(layer, mesh: Mesh, cluster_axis: int):
    cluster = "_axis_0" if cluster_axis == 0 else "_axis_1"
    compound = ("_axis_0", "_axis_1")

    attn = layer.self_attn
    xs.mark_sharding(attn.q_proj.weight, mesh, (cluster, None))
    xs.mark_sharding(attn.k_proj.weight, mesh, (cluster, None))
    xs.mark_sharding(attn.v_proj.weight, mesh, (cluster, None))
    xs.mark_sharding(attn.o_proj.weight, mesh, (None, cluster))
    if attn.q_proj.bias is not None:
        xs.mark_sharding(attn.q_proj.bias, mesh, (cluster,))
        xs.mark_sharding(attn.k_proj.bias, mesh, (cluster,))
        xs.mark_sharding(attn.v_proj.bias, mesh, (cluster,))

    xs.mark_sharding(layer.input_layernorm.weight, mesh, (cluster,))
    xs.mark_sharding(layer.post_attention_layernorm.weight, mesh, (cluster,))

    mlp = layer.mlp  # A2aSparseMLP
    xs.mark_sharding(mlp.experts.gate_proj, mesh, (compound, None, None))
    xs.mark_sharding(mlp.experts.up_proj, mesh, (compound, None, None))
    xs.mark_sharding(mlp.experts.down_proj, mesh, (compound, None, None))
    if mlp.experts.gate_proj_bias is not None:
        xs.mark_sharding(mlp.experts.gate_proj_bias, mesh, (compound, None))
        xs.mark_sharding(mlp.experts.up_proj_bias, mesh, (compound, None))
        xs.mark_sharding(mlp.experts.down_proj_bias, mesh, (compound, None))
    xs.mark_sharding(mlp.router.weight, mesh, (None, cluster))


@pytest.mark.push
@pytest.mark.llmbox
def test_gpt_oss_layer_bwd_pcc_2x4():
    """
    Attention sharded 2-way on axis_0, experts compound-sharded across both
    mesh axes (built via build_expert_mapping with mesh_shape=(2,4)). Input
    is sharded ``("_axis_1", None, "_axis_0")`` following the same pattern as
    the deepseek_v3_2 / glm4 2x4 MoE tests.
    """
    mesh, device, num_devices = _setup_mesh()
    cluster_axis = 0
    batch_size, seq_len = 8, 32

    torch.manual_seed(0)
    layer_cpu, config = _build_layer(num_experts=num_devices)
    torch.manual_seed(0)
    layer_tt, _ = _build_layer(num_experts=num_devices)
    layer_tt.load_state_dict(layer_cpu.state_dict())

    for layer in (layer_tt, layer_cpu):
        enable_sparse_mlp(
            layer,
            mesh=MESH_SHAPE,
            cluster_axis=cluster_axis,
            config=config,
            use_dense_matmul=True,
            deinterleave_fused_experts=True,
        )

    hidden_states = torch.randn(
        (batch_size, seq_len, config.hidden_size), dtype=torch.float32
    )
    position_ids = torch.arange(seq_len).unsqueeze(0).repeat(batch_size, 1)
    rotary_dim = config.head_dim // 2  # gpt-oss splits head_dim in half for rotary
    cos = torch.randn(batch_size, seq_len, rotary_dim, dtype=torch.float32)
    sin = torch.randn(batch_size, seq_len, rotary_dim, dtype=torch.float32)

    x_cpu = hidden_states.detach().clone().requires_grad_(True)
    out_cpu = layer_cpu(
        x_cpu,
        attention_mask=None,
        position_ids=position_ids,
        past_key_values=None,
        use_cache=False,
        cache_position=None,
        position_embeddings=(cos, sin),
    )
    out_cpu.sum().backward()

    layer_tt = layer_tt.to(device)
    _shard_decoder_layer(layer_tt, mesh, cluster_axis)

    x_tt = hidden_states.detach().clone().to(device).requires_grad_(True)
    xs.mark_sharding(x_tt, mesh, ("_axis_1", None, "_axis_0"))
    cos_tt, sin_tt = cos.to(device), sin.to(device)
    pos_tt = position_ids.to(device)

    out_tt = layer_tt(
        x_tt,
        attention_mask=None,
        position_ids=pos_tt,
        past_key_values=None,
        use_cache=False,
        cache_position=None,
        position_embeddings=(cos_tt, sin_tt),
    )
    out_tt.sum().backward()
    torch_xla.sync()

    pcc = 0.95
    cases = {
        "out": (out_cpu, out_tt.cpu()),
        "dx": (x_cpu.grad, x_tt.grad.cpu()),
        "gate_proj.grad": (
            layer_cpu.mlp.experts.gate_proj.grad,
            layer_tt.mlp.experts.gate_proj.grad.cpu(),
        ),
        "up_proj.grad": (
            layer_cpu.mlp.experts.up_proj.grad,
            layer_tt.mlp.experts.up_proj.grad.cpu(),
        ),
        "down_proj.grad": (
            layer_cpu.mlp.experts.down_proj.grad,
            layer_tt.mlp.experts.down_proj.grad.cpu(),
        ),
        "router.grad": (
            layer_cpu.mlp.router.weight.grad,
            layer_tt.mlp.router.weight.grad.cpu(),
        ),
        "q_proj.grad": (
            layer_cpu.self_attn.q_proj.weight.grad,
            layer_tt.self_attn.q_proj.weight.grad.cpu(),
        ),
        "o_proj.grad": (
            layer_cpu.self_attn.o_proj.weight.grad,
            layer_tt.self_attn.o_proj.weight.grad.cpu(),
        ),
    }
    for name, (g, t) in cases.items():
        print(f"[PCC] {name}", flush=True)
        compute_pcc(g, t, required_pcc=pcc)
