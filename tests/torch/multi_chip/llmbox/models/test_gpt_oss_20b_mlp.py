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
from transformers.models.gpt_oss.modeling_gpt_oss import (
    GptOssExperts,
    GptOssMLP,
    GptOssRMSNorm,
    GptOssTopKRouter,
)

from third_party.tt_forge_models.gpt_oss.pytorch.loader import ModelLoader, ModelVariant


def gpt_oss():
    # setup_spmd()

    # Connect the device and create an xla mesh.
    # device: torch.device = torch_xla.device()
    # mesh: Mesh = create_device_mesh()
    breakpoint()
    config = AutoConfig.from_pretrained(
        self._variant_config.pretrained_model_name, trust_remote_code=True
    )
    loader = ModelLoader(variant=None)
    config = loader.load_config()
    mlp = GptOssMLP(config).to(torch.bfloat16)
    mlp.eval()

    inputs = loader.load_inputs()
    batch_size = inputs["input_ids"].shape[0]
    seq_len = inputs["input_ids"].shape[1]
    hidden_states = torch.randn(
        (batch_size, seq_len, config.hidden_size), dtype=torch.bfloat16
    )
    rmsnorm = GptOssRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
    hidden_states = rmsnorm(hidden_states)

    with torch.no_grad():
        output = mlp(hidden_states)
    breakpoint()
    print("Output logits:", output)


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
    # xr.set_device_type("TT")
    torch_xla.sync()
    torch._dynamo.reset()
    gpt_oss()
