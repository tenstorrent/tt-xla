# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
import numpy as np
import torch
import torch_xla
import torch_xla.core.xla_model as xm
import torch_xla.runtime as xr
import torch_xla.distributed.spmd as xs
from torch_xla.distributed.spmd import Mesh
import os
import pytest

from transformers.models.llama.modeling_llama import LlamaAttention, LlamaRotaryEmbedding, LlamaMLP, LlamaDecoderLayer, LlamaModel
from transformers.models.llama.configuration_llama import LlamaConfig
# needs to be set at module level to unsure it gets picked up before torch-xla C++ code is initialized
os.environ["DISABLE_NUMERIC_CC_TOKEN"] = "1"
def setup_tt_environment():
    """Setup TensorTrent environment and plugin."""
    os.environ["PJRT_DEVICE"] = "TT"
    os.environ["XLA_STABLEHLO_COMPILE"] = "1"
    os.environ["XLA_ALWAYS_ALLREDUCE"] = "1"
    os.environ["ENABLE_AUTO_PARALLEL"] = "TRUE"
    os.environ["CONVERT_SHLO_TO_SHARDY"] = "1"
    os.environ["MESH_SHAPE"] = "1,8"
    os.environ["LOGGER_LEVEL"] = "DEBUG"

    from torch_xla.experimental import plugins

    class TTPjrtPlugin(plugins.DevicePlugin):
        def library_path(self):
            return os.path.join(
                os.path.dirname(__file__), "../../build/src/tt/pjrt_plugin_tt.so"
            )

    plugins.register_plugin("TT", TTPjrtPlugin())
    xr.use_spmd()
    torch_xla.sync(True, True)


def create_mesh():
    """Create device mesh for testing."""
    num_devices = xr.global_runtime_device_count()
    mesh_shape = (1, 8)
    device_ids = np.array(range(num_devices))
    return Mesh(device_ids, mesh_shape, ("batch", "model"))

def compute_pcc(x: torch.Tensor, y: torch.Tensor):
    x_flat, y_flat = x.flatten(), y.flatten()
    vx, vy = x_flat - x_flat.mean(), y_flat - y_flat.mean()
    denom = vx.norm() * vy.norm()
    return torch.tensor(float("nan")) if denom == 0 else (vx @ vy) / denom

def test_mlp():
    setup_tt_environment()
    mesh = create_mesh()
    
    B = 1
    S = 1024
    H = 8192
    config = LlamaConfig.from_pretrained("meta-llama/Meta-Llama-3-70B")
    mlp = LlamaMLP(config)

    hidden_states = torch.randn(B, S, H)
    out_cpu = mlp(hidden_states)
    hidden_states = hidden_states.to(torch_xla.device())
    xs.mark_sharding(hidden_states, mesh, (None, None, None))

    mlp = mlp.to(torch_xla.device())
    xs.mark_sharding(mlp.up_proj.weight, mesh, ("model", None))
    xs.mark_sharding(mlp.gate_proj.weight, mesh, ("model", None))
    xs.mark_sharding(mlp.down_proj.weight, mesh, (None, "model"))

    out = mlp(hidden_states)
    out = out.cpu()
    pcc = compute_pcc(out, out_cpu)
    assert pcc > 0.99


def test_decode_layer():
    setup_tt_environment()
    mesh = create_mesh()
    
    B = 1
    S = 1024
    H = 8192
    config = LlamaConfig.from_pretrained("meta-llama/Meta-Llama-3-70B")
    layer = LlamaDecoderLayer(config, 0)

    hidden_states = torch.randn(B, S, H)
    position_ids = torch.arange(0, S).unsqueeze(0)
    rot_emb = LlamaRotaryEmbedding(config)
    (cos, sin) = rot_emb(hidden_states, position_ids)
    out_cpu = layer(hidden_states, attention_mask=None, position_embeddings=(cos, sin))

    hidden_states = hidden_states.to(torch_xla.device())
    xs.mark_sharding(hidden_states, mesh, (None, None, None))

    layer = layer.to(torch_xla.device())
    xs.mark_sharding(layer.mlp.up_proj.weight, mesh, ("model", None))
    xs.mark_sharding(layer.mlp.gate_proj.weight, mesh, ("model", None))
    xs.mark_sharding(layer.mlp.down_proj.weight, mesh, (None, "model"))

    xs.mark_sharding(layer.self_attn.q_proj.weight, mesh, ("model", None))
    xs.mark_sharding(layer.self_attn.k_proj.weight, mesh, ("model", None))
    xs.mark_sharding(layer.self_attn.v_proj.weight, mesh, ("model", None))
    xs.mark_sharding(layer.self_attn.o_proj.weight, mesh, (None, "model"))

    out = layer(hidden_states, attention_mask=None, position_embeddings=(cos, sin))
    out = out[0].cpu()
    pcc = compute_pcc(out, out_cpu[0])
    assert pcc > 0.99

def test_llama_attention():
    setup_tt_environment()
    mesh = create_mesh()
    
    B = 1
    S = 1024
    H = 8192
    config = LlamaConfig.from_pretrained("meta-llama/Meta-Llama-3-70B")
    attention = LlamaAttention(config, 0)
    hidden_states = torch.randn(B, S, H)
    position_ids = torch.arange(0, S).unsqueeze(0)
    rot_emb = LlamaRotaryEmbedding(config)
    (cos, sin) = rot_emb(hidden_states, position_ids)
    out_cpu = attention(hidden_states, (cos, sin), attention_mask=None)

    hidden_states = hidden_states.to(torch_xla.device())
    xs.mark_sharding(hidden_states, mesh, (None, None, None))

    cos = cos.to(torch_xla.device())
    xs.mark_sharding(cos, mesh, (None, None, None))
    sin = sin.to(torch_xla.device())
    xs.mark_sharding(sin, mesh, (None, None, None))
    
    attention = attention.to(torch_xla.device())
    xs.mark_sharding(attention.q_proj.weight, mesh, ("model", None))
    xs.mark_sharding(attention.k_proj.weight, mesh, ("model", None))
    xs.mark_sharding(attention.v_proj.weight, mesh, ("model", None))
    xs.mark_sharding(attention.o_proj.weight, mesh, (None, "model"))

    out = attention(hidden_states, (cos, sin), attention_mask=None)
    out = out[0].cpu()
    pcc = compute_pcc(out, out_cpu[0])
    assert pcc > 0.99
    
    print(f"Attention output shape: {out.shape}")
    print("Attention test completed!")


def test_basic_attention():
    setup_tt_environment()
    mesh = create_mesh()
    config = LlamaConfig.from_pretrained("meta-llama/Meta-Llama-3-70B")
    attention = LlamaAttention(config, 0)

    B = 1
    S = 1024
    H = 8192
    KV = 1024
    rep = H//KV
    head_dim = 128
    scale = head_dim**-0.5
    
    hidden_states = torch.randn(B, S, H)
    query_weight = attention.q_proj.weight
    key_weight =   attention.k_proj.weight
    val_weight =   attention.v_proj.weight
    out_weight =   attention.o_proj.weight

    def attention(hidden_states, query_weight, key_weight, val_weight, out_weight):
        q_proj = torch.matmul(hidden_states, query_weight.transpose(-1, -2))
        q_proj = q_proj.view(B, S, -1, head_dim).transpose(1,2)

        k_proj = torch.matmul(hidden_states, key_weight.transpose(-1, -2))
        k_proj = k_proj.view(B, S, -1, head_dim).transpose(1,2)
        k_proj = torch.repeat_interleave(k_proj, rep, 1)

        v_proj = torch.matmul(hidden_states, val_weight.transpose(-1, -2))
        v_proj = v_proj.view(B, S, -1, head_dim).transpose(1,2)
        v_proj = torch.repeat_interleave(v_proj, rep, 1)
        
        attn_weights = torch.matmul(q_proj, k_proj.transpose(2, 3))
        attn_weights = attn_weights * scale
        attn_weights = torch.nn.functional.softmax(attn_weights, dim=-1)

        attn_output = torch.matmul(attn_weights, v_proj)
        attn_output = attn_output.transpose(1, 2).reshape(B, S, H)

        attn_output = torch.matmul(attn_output, out_weight.transpose(-1, -2))
        return attn_output

    cpu_out = attention(hidden_states, query_weight, key_weight, val_weight, out_weight)

    hidden_states = hidden_states.to(torch_xla.device())
    xs.mark_sharding(hidden_states, mesh, (None, None, None))

    query_weight = query_weight.to(torch_xla.device())
    xs.mark_sharding(query_weight, mesh, ("model", None))

    key_weight = key_weight.to(torch_xla.device())
    xs.mark_sharding(key_weight, mesh, ("model", None))

    val_weight = val_weight.to(torch_xla.device())
    xs.mark_sharding(val_weight, mesh, ("model", None))

    out_weight = out_weight.to(torch_xla.device())
    xs.mark_sharding(out_weight, mesh, (None, "model"))

    dev_out = attention(hidden_states, query_weight, key_weight, val_weight, out_weight)
    dev_out = dev_out.cpu()

    pcc = compute_pcc(dev_out, cpu_out)
    assert pcc > 0.99

def test_llama():
    setup_tt_environment()
    mesh = create_mesh()

    B = 1
    S = 1024
    H = 8192
    # config = LlamaConfig.from_pretrained("meta-llama/Meta-Llama-3-70B")
    config = LlamaConfig.from_pretrained("meta-llama/Meta-Llama-3.1-8B")
    # config.num_hidden_layers = 30
    llama = LlamaModel(config)

    input_ids = torch.randint(0, config.vocab_size, (B, S))
    out_cpu = llama(input_ids=input_ids, attention_mask=None)
    llama = llama.to(torch.bfloat16)
    input_ids = input_ids.to("xla")
    llama = llama.to("xla")

    # print(f"[HET DEBUG] Number of layers: {len(llama.layers)}")
    # for end in range(len(llama.layers)):
    #     for layer in llama.layers[:end+1]:
    #         xs.mark_sharding(layer.mlp.up_proj.weight, mesh, ("model", None))
    #         xs.mark_sharding(layer.mlp.gate_proj.weight, mesh, ("model", None))
    #         xs.mark_sharding(layer.mlp.down_proj.weight, mesh, (None, "model"))

    #         xs.mark_sharding(layer.self_attn.q_proj.weight, mesh, ("model", None))
    #         xs.mark_sharding(layer.self_attn.k_proj.weight, mesh, ("model", None))
    #         xs.mark_sharding(layer.self_attn.v_proj.weight, mesh, ("model", None))
    #         xs.mark_sharding(layer.self_attn.o_proj.weight, mesh, (None, "model"))
    #     out = llama(input_ids=input_ids, attention_mask=None)
    #     out = out.last_hidden_state.cpu().float()
    #     pcc = compute_pcc(out, out_cpu.last_hidden_state)
    #     print(f"LLAMA PCC after {end+1} layers: {pcc}")

    for layer in llama.layers:
        xs.mark_sharding(layer.mlp.up_proj.weight, mesh, ("model", None))
        xs.mark_sharding(layer.mlp.gate_proj.weight, mesh, ("model", None))
        xs.mark_sharding(layer.mlp.down_proj.weight, mesh, (None, "model"))

        xs.mark_sharding(layer.self_attn.q_proj.weight, mesh, ("model", None))
        xs.mark_sharding(layer.self_attn.k_proj.weight, mesh, ("model", None))
        xs.mark_sharding(layer.self_attn.v_proj.weight, mesh, ("model", None))
        xs.mark_sharding(layer.self_attn.o_proj.weight, mesh, (None, "model"))

    out = llama(input_ids=input_ids, attention_mask=None)
    out = out.last_hidden_state.cpu().float()
    pcc = compute_pcc(out, out_cpu.last_hidden_state)
    print(f"LLAMA PCC: {pcc}")
    # assert pcc > 0.95

def test_transpose_weight():
    setup_tt_environment()
    mesh = create_mesh()

    B = 1
    S = 1024
    H = 8192
    KV = 1024
    rep = H//KV
    head_dim = 128
    scale = head_dim**-0.5
    
    hidden_states = torch.randn(B, S, H)
    query_weight = torch.randn(H, H)

    hidden_states = hidden_states.to(torch_xla.device())
    xs.mark_sharding(hidden_states, mesh, (None, None, None))

    query_weight = query_weight.to(torch_xla.device())
    xs.mark_sharding(query_weight, mesh, ("model", None))

    out = torch.matmul(hidden_states, query_weight.transpose(-1, -2))
    out = out.cpu()

if __name__ == "__main__":
    # test_basic_attention()
    test_llama()
    print("All tests passed!")