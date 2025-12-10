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

from third_party.tt_forge_models.gpt_oss.pytorch.loader import ModelLoader, ModelVariant


def gpt_oss():
    setup_spmd()

    # Connect the device and create an xla mesh.
    device: torch.device = torch_xla.device()
    mesh: Mesh = create_device_mesh()

    loader = ModelLoader(variant=None)
    model = loader.load_model()
    inputs = loader.load_inputs()

    # Move model and inputs to xla device.
    model = model.to(device)
    inputs["input_ids"] = inputs["input_ids"].to(device)
    inputs["attention_mask"] = inputs["attention_mask"].to(device)

    mark_sharding_on_inputs_and_model(model, mesh)

    with torch.no_grad():
        output = model(**inputs)

    output_logits = output.logits.to("cpu")
    print("Output logits:", output_logits)
    breakpoint()
    print("gpt-oss test completed successfully.")


def mark_sharding_on_inputs_and_model(model: torch.nn.Module, mesh: Mesh):

    print("Applying tensor parallel sharding to base model")

    # Apply replication to each transformer layer
    for layer in model.model.layers:

        # q_proj weight shape: [2880, 4096]
        # Sharded colwise: [2880, 4096/num_devices]
        xs.mark_sharding(layer.self_attn.q_proj.weight, mesh, ("model", None))

        # k_proj weight shape: [2880, 512]
        # Sharded colwise: [2880, 512/num_devices]
        xs.mark_sharding(layer.self_attn.k_proj.weight, mesh, ("model", None))

        # v_proj weight shape: [2880, 512]
        # Sharded colwise: [2880, 512/num_devices]
        xs.mark_sharding(layer.self_attn.v_proj.weight, mesh, ("model", None))

        # o_proj weight shape: [4096, 2880]
        # Sharded rowwise: [4096/num_devices, 2880]
        xs.mark_sharding(layer.self_attn.o_proj.weight, mesh, (None, "model"))

        # sinks shape: [4096] -> local. rowwise
        xs.mark_sharding(layer.self_attn.sinks, mesh, ("model",))

        xs.mark_sharding(layer.mlp.router.weight, mesh, (None, "model"))

        # Shard experts across devices: 32 / 8 ->. 4 expert per device
        xs.mark_sharding(layer.mlp.experts.gate_up_proj, mesh, ("model", None, None))
        xs.mark_sharding(layer.mlp.experts.gate_up_proj_bias, mesh, ("model", None))
        xs.mark_sharding(layer.mlp.experts.down_proj, mesh, ("model", None, None))
        xs.mark_sharding(layer.mlp.experts.down_proj_bias, mesh, ("model", None))


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
