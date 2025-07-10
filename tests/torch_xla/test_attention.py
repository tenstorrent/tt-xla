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

from transformers.models.llama.modeling_llama import LlamaAttention, LlamaRotaryEmbedding
from transformers.models.llama.configuration_llama import LlamaConfig
# needs to be set at module level to unsure it gets picked up before torch-xla C++ code is initialized
os.environ["DISABLE_NUMERIC_CC_TOKEN"] = "1"
def setup_tt_environment():
    """Setup TensorTrent environment and plugin."""
    os.environ["PJRT_DEVICE"] = "TT"
    os.environ["XLA_STABLEHLO_COMPILE"] = "1"
    os.environ["XLA_ALWAYS_ALLREDUCE"] = "1"
    os.environ["ENABLE_AUTO_PARALLEL"] = "TRUE"
    os.environ["MESH_SHAPE"] = "2,4"
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

def test_llama_attention():
    # Get pretrained config from meta-llama/Meta-Llama-3-70B
    setup_tt_environment()
    mesh = create_mesh()
    
    B = 1
    S = 1024
    H = 8192
    config = LlamaConfig.from_pretrained("meta-llama/Meta-Llama-3-70B")
    attention = LlamaAttention(config, 0)
    position_ids = torch.arange(0, S).unsqueeze(0)
    rot_emb = LlamaRotaryEmbedding(config)
    hidden_states = torch.randn(B, S, H)
    (cos, sin) = rot_emb(hidden_states, position_ids)

    hidden_states = hidden_states.to(torch_xla.device())
    xs.mark_sharding(hidden_states, mesh, (None, None, None))

    cos = cos.to(torch_xla.device())
    xs.mark_sharding(cos, mesh, (None, None, None))
    sin = sin.to(torch_xla.device())
    xs.mark_sharding(sin, mesh, (None, None, None))
    
    attention = attention.to(torch_xla.device())
    xs.mark_sharding(attention.q_proj.weight, mesh, (None, "model"))
    xs.mark_sharding(attention.k_proj.weight, mesh, (None, "model"))
    xs.mark_sharding(attention.v_proj.weight, mesh, (None, "model"))
    xs.mark_sharding(attention.o_proj.weight, mesh, ("model", None))

    out = attention(hidden_states, (cos, sin), attention_mask=None)
    out = out[0].cpu()
    
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
    query_weight = attention.q_proj.weight.transpose(-1, -2)
    key_weight =   attention.k_proj.weight.transpose(-1, -2)
    val_weight =   attention.v_proj.weight.transpose(-1, -2)
    out_weight =   attention.o_proj.weight.transpose(-1, -2)

    def attention(hidden_states, query_weight, key_weight, val_weight, out_weight):
        q_proj = torch.matmul(hidden_states, query_weight)
        q_proj = q_proj.view(B, S, -1, head_dim).transpose(1,2)

        k_proj = torch.matmul(hidden_states, key_weight)
        k_proj = k_proj.view(B, S, -1, head_dim).transpose(1,2)
        k_proj = torch.repeat_interleave(k_proj, rep, 1)

        v_proj = torch.matmul(hidden_states, val_weight)
        v_proj = v_proj.view(B, S, -1, head_dim).transpose(1,2)
        v_proj = torch.repeat_interleave(v_proj, rep, 1)
        
        attn_weights = torch.matmul(q_proj, k_proj.transpose(2, 3))
        attn_weights = attn_weights * scale
        attn_weights = torch.nn.functional.softmax(attn_weights, dim=-1)

        attn_output = torch.matmul(attn_weights, v_proj)
        attn_output = attn_output.transpose(1, 2).reshape(B, S, H)

        attn_output = torch.matmul(attn_output, out_weight)
        return attn_output

    cpu_out = attention(hidden_states, query_weight, key_weight, val_weight, out_weight)

    hidden_states = hidden_states.to(torch_xla.device())
    xs.mark_sharding(hidden_states, mesh, (None, "batch", None))

    query_weight = query_weight.to(torch_xla.device())
    xs.mark_sharding(query_weight, mesh, (None, "model"))

    key_weight = key_weight.to(torch_xla.device())
    xs.mark_sharding(key_weight, mesh, (None, "model"))

    val_weight = val_weight.to(torch_xla.device())
    xs.mark_sharding(val_weight, mesh, (None, "model"))

    out_weight = out_weight.to(torch_xla.device())
    xs.mark_sharding(out_weight, mesh, ("model", None))

    dev_out = attention(hidden_states, query_weight, key_weight, val_weight, out_weight)
    dev_out = dev_out.cpu()

    assert torch.allclose(dev_out, cpu_out, atol=0.02)


if __name__ == "__main__":
    test_basic_attention()
    print("All tests passed!")
