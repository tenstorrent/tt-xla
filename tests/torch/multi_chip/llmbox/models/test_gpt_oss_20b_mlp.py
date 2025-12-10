# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import os

import numpy as np
import pytest
import torch
import torch_xla
import torch_xla.core.xla_model as xm
import torch_xla.distributed.spmd as xs
import torch_xla.runtime as xr
from torch_xla.distributed.spmd import Mesh
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
from transformers.models.gpt_oss.modeling_gpt_oss import (
    GptOssExperts,
    GptOssMLP,
    GptOssRMSNorm,
    GptOssTopKRouter,
)

from third_party.tt_forge_models.gpt_oss.pytorch.loader import ModelLoader, ModelVariant


def gpt_oss():
    setup_spmd()

    # Connect the device and create an xla mesh.
    device: torch.device = torch_xla.device()
    mesh: Mesh = create_device_mesh()

    loader = ModelLoader(variant=ModelVariant.GPT_OSS_20B, num_layers=1)
    model = loader.load_model()
    config = loader.load_config()
    inputs = loader.load_inputs()
    batch_size = inputs["input_ids"].shape[0]  # 1
    seq_len = inputs["input_ids"].shape[1]  # 71

    hidden_states = torch.randn(
        (batch_size, seq_len, config.hidden_size), dtype=torch.bfloat16
    )
    mlp: GptOssMLP = model.model.layers[0].mlp
    mlp = mlp.to(device)
    hidden_states = hidden_states.to(device)

    mark_sharding_on_inputs_and_model(mlp, mesh)

    with torch.no_grad():
        output = mlp(hidden_states)

    print("MLP output", output[0].to("cpu"))


def mark_sharding_on_inputs_and_model(mlp, mesh):
    print("Applying tensor parallel sharding to mlp")
    xs.mark_sharding(mlp.router.weight, mesh, (None, None))
    xs.mark_sharding(mlp.router.bias, mesh, (None,))

    # Shard experts across devices: 32 / 8 ->. 4 expert per device
    xs.mark_sharding(mlp.experts.gate_up_proj, mesh, ("model", None, None))
    xs.mark_sharding(mlp.experts.gate_up_proj_bias, mesh, ("model", None))
    xs.mark_sharding(mlp.experts.down_proj, mesh, ("model", None, None))
    xs.mark_sharding(mlp.experts.down_proj_bias, mesh, ("model", None))


def setup_spmd():
    """
    Initializes SPMD mode in torch_xla.
    """

    print("Setting up XLA environment...")

    # Converts the StableHLO emitted by torch-xla to the Shardy dialect
    os.environ["CONVERT_SHLO_TO_SHARDY"] = "1"

    # Initialize SPMD
    xr.use_spmd()
    print("XLA environment configured.")


def create_device_mesh() -> Mesh:
    """
    Create device mesh for tensor parallelism.

    Returns:
        Mesh object for SPMD operations
    """
    num_devices = xr.global_runtime_device_count()
    mesh_shape = (1, num_devices)
    device_ids = np.array(range(num_devices))
    mesh = Mesh(device_ids, mesh_shape, ("batch", "model"))
    print(f"Created device mesh: {mesh_shape} with {num_devices} devices")
    return mesh


if __name__ == "__main__":
    # By default torch_xla uses the CPU device so we have to set it to TT device.
    xr.set_device_type("TT")
    torch._dynamo.reset()
    gpt_oss()
