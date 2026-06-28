# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
Tensor-parallel inference example for the SRPO (tencent/SRPO) denoiser.

SRPO is a FLUX.1-dev ``FluxTransformer2DModel`` fine-tuned with Semantic
Relative Preference Optimization (arXiv:2509.06942). At native 1024x1024 the
11.9B bf16 denoiser (~24 GB) plus its activations does not fit a single 32 GB
Blackhole chip, so it is run **tensor-parallel** across the available devices.

This script demonstrates a single FLUX/SRPO denoising step (one transformer
forward) sharded Megatron column->row across the mesh, mirroring how the
bring-up validated the model on device. The sharding plan, mesh, and inputs all
come from the tt-forge-models ``ModelLoader`` public API:

  - ``loader.get_mesh_config(num_devices)`` -> (mesh_shape, mesh_names)
  - ``loader.load_shard_spec(model)``       -> {param_tensor: partition_spec}
  - ``loader.load_inputs()``                -> the FLUX transformer inputs

Attention q/k/v and MLP up-projections are column-parallel; the attention/MLP
output projections and the AdaLN modulation linears are row-parallel (the latter
keeps a downstream activation concat within Blackhole L1 at native resolution).
"""

import os

import numpy as np
import torch
import torch_xla
import torch_xla.distributed.spmd as xs
import torch_xla.runtime as xr
from torch_xla.distributed.spmd import Mesh

from third_party.tt_forge_models.srpo.pytorch import ModelLoader, ModelVariant


# --------------------------------
# SRPO tensor-parallel example
# --------------------------------
def srpo_tp():
    # Enable SPMD (Shardy) sharding and discover the device mesh.
    os.environ["CONVERT_SHLO_TO_SHARDY"] = "1"
    xr.use_spmd()

    num_devices = xr.global_runtime_device_count()

    # Build the SRPO denoiser and its native-resolution inputs via the loader.
    loader = ModelLoader(ModelVariant.SRPO)
    model = loader.load_model(dtype_override=torch.bfloat16).eval()
    inputs = loader.load_inputs(dtype_override=torch.bfloat16)

    # Mesh from the loader's own contract: ("batch", "model") with all devices
    # on the model axis (batch axis = 1, tensor-parallel only).
    mesh_shape, mesh_names = loader.get_mesh_config(num_devices)
    device_ids = np.array(range(num_devices))
    mesh = Mesh(device_ids, mesh_shape, mesh_names)

    # Move model and inputs to the TT device.
    device = torch_xla.device()
    model = model.to(device)
    inputs = _inputs_to_device(inputs, device)

    # Apply the loader's Megatron column->row tensor-parallel shard spec.
    shard_specs = loader.load_shard_spec(model)
    for tensor, partition_spec in shard_specs.items():
        xs.mark_sharding(tensor, mesh, partition_spec)

    # Compile and run a single denoising step on device.
    compiled_model = torch.compile(model, backend="tt")
    with torch.no_grad():
        output = compiled_model(**inputs)

    # FluxTransformer2DModel returns the predicted velocity as `.sample`.
    return output.sample.cpu().float()


def _inputs_to_device(inputs, device):
    """Move the FLUX transformer input tensors to ``device``.

    ``load_inputs`` returns a dict of tensors plus a plain ``joint_attention_kwargs``
    dict; move the tensors and leave the non-tensor entries untouched.
    """
    moved = {}
    for key, value in inputs.items():
        if isinstance(value, torch.Tensor):
            moved[key] = value.to(device)
        else:
            moved[key] = value
    return moved


def post_process_output(velocity):
    """Print a human-readable summary of the predicted denoiser output."""
    print("SRPO denoiser (1 tensor-parallel step) — predicted velocity:")
    print(f"  shape : {tuple(velocity.shape)}")
    print(f"  dtype : {velocity.dtype}")
    print(f"  finite: {bool(torch.isfinite(velocity).all())}")
    print(
        f"  stats : min={velocity.min():.4f} "
        f"max={velocity.max():.4f} mean={velocity.mean():.4f} "
        f"std={velocity.std():.4f}"
    )


def test_srpo_tp():
    """SRPO denoiser runs tensor-parallel and produces a sensible velocity field.

    Checks that the device output is finite and keeps the packed-latent shape
    ``(batch, num_patches, latent_channels)`` of the input, so the example
    guards a real end-to-end denoising step rather than just a compile.
    """
    xr.set_device_type("TT")

    velocity = srpo_tp()

    assert torch.isfinite(velocity).all(), "SRPO output contains non-finite values"
    # Native 1024x1024 packs to 4096 latent patches of 64 channels each.
    assert velocity.shape == (1, 4096, 64), f"unexpected output shape {velocity.shape}"

    post_process_output(velocity)
    print("SRPO tensor-parallel denoising step produced a valid velocity field.")


if __name__ == "__main__":
    # By default torch_xla uses the CPU device so we have to set it to TT device.
    xr.set_device_type("TT")

    velocity = srpo_tp()
    post_process_output(velocity)
