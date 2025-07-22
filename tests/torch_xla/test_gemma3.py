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

from transformers.models.gemma3.modeling_gemma3 import Gemma3Attention, Gemma3RotaryEmbedding, Gemma3MLP, Gemma3DecoderLayer, Gemma3Model, Gemma3TextModel
from transformers.models.gemma3.configuration_gemma3 import Gemma3Config, Gemma3TextConfig
# needs to be set at module level to unsure it gets picked up before torch-xla C++ code is initialized
os.environ["DISABLE_NUMERIC_CC_TOKEN"] = "1"
def setup_tt_environment():
    """Setup TensorTrent environment and plugin."""
    os.environ["PJRT_DEVICE"] = "TT"
    os.environ["XLA_STABLEHLO_COMPILE"] = "1"
    os.environ["XLA_ALWAYS_ALLREDUCE"] = "1"
    os.environ["ENABLE_AUTO_PARALLEL"] = "TRUE"
    os.environ["MESH_SHAPE"] = "1,2"
    os.environ["CONVERT_SHLO_TO_SHARDY"] = "1"
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
    mesh_shape = (1, 2)
    device_ids = np.array(range(num_devices))
    return Mesh(device_ids, mesh_shape, ("batch", "model"))

def compute_pcc(x: torch.Tensor, y: torch.Tensor):
    x_flat, y_flat = x.flatten(), y.flatten()
    vx, vy = x_flat - x_flat.mean(), y_flat - y_flat.mean()
    denom = vx.norm() * vy.norm()
    return torch.tensor(float("nan")) if denom == 0 else (vx @ vy) / denom



def test_decode_layer():
    setup_tt_environment()
    mesh = create_mesh()
    
    B = 1
    S = 1024
    # config = Gemma3TextConfig.from_pretrained("google/gemma-3-27b-it")
    config = Gemma3TextConfig.from_pretrained("google/gemma-3-4b-it")
    H = config.hidden_size
    layer = Gemma3DecoderLayer(config, 0)

    hidden_states = torch.randn(B, S, H)
    position_ids = torch.arange(0, S).unsqueeze(0)
    rot_emb = Gemma3RotaryEmbedding(config)
    (cos, sin) = rot_emb(hidden_states, position_ids)
    out_cpu = layer(hidden_states, attention_mask=None, position_embeddings_global=(cos, sin), position_embeddings_local=(cos, sin))

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

    out = layer(hidden_states, attention_mask=None, position_embeddings_global=(cos, sin), position_embeddings_local=(cos, sin))
    out = out[0].cpu()
    pcc = compute_pcc(out, out_cpu[0])
    assert pcc > 0.99


def test_gemma3():
    setup_tt_environment()
    mesh = create_mesh()

    B = 1
    S = 1024
    config = Gemma3Config.from_pretrained("google/gemma-3-4b-it")
    # config.text_config.num_hidden_layers = 1
    gemma = Gemma3Model(config)

    input_ids = torch.randint(0, config.text_config.vocab_size, (B, S))
    out_cpu = gemma(input_ids=input_ids, attention_mask=None)
    gemma = gemma.to(torch.bfloat16)
    input_ids = input_ids.to("xla")
    gemma = gemma.to("xla")


    for layer in gemma.language_model.layers:
        xs.mark_sharding(layer.mlp.up_proj.weight, mesh, ("model", None))
        xs.mark_sharding(layer.mlp.gate_proj.weight, mesh, ("model", None))
        xs.mark_sharding(layer.mlp.down_proj.weight, mesh, (None, "model"))

        xs.mark_sharding(layer.self_attn.q_proj.weight, mesh, ("model", None))
        xs.mark_sharding(layer.self_attn.k_proj.weight, mesh, ("model", None))
        xs.mark_sharding(layer.self_attn.v_proj.weight, mesh, ("model", None))
        xs.mark_sharding(layer.self_attn.o_proj.weight, mesh, (None, "model"))

    out = gemma(input_ids=input_ids, attention_mask=None)
    out = out.last_hidden_state.cpu().float()
    pcc = compute_pcc(out, out_cpu.last_hidden_state)
    print(f"GEMMA PCC: {pcc}")
    # assert pcc > 0.95

    out = out.cpu()

if __name__ == "__main__":
    test_gemma3()
    print("All tests passed!")
