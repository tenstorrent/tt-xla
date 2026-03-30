# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
Simple SPMD example: replicate a 1024x1024 bf16 tensor of ones across a 4x16
mesh and add 1 to it.

Usage:
    python examples/pytorch/spmd_replicate_add.py
"""

import os

import numpy as np
import torch
import torch_xla
import torch_xla.distributed.spmd as xs
import torch_xla.runtime as xr
from torch_xla.distributed.spmd import Mesh


def setup_spmd():
    os.environ["CONVERT_SHLO_TO_SHARDY"] = "1"
    xr.use_spmd()


def create_mesh(mesh_shape=(4, 16)) -> Mesh:
    num_devices = xr.global_runtime_device_count()
    assert num_devices == mesh_shape[0] * mesh_shape[1], (
        f"Expected {mesh_shape[0] * mesh_shape[1]} devices for a {mesh_shape} mesh, "
        f"but found {num_devices}."
    )
    device_ids = np.arange(num_devices)
    return Mesh(device_ids, mesh_shape, ("x", "y"))


def main():
    xr.set_device_type("TT")
    setup_spmd()

    device = torch_xla.device()
    mesh = create_mesh(mesh_shape=(4, 16))

    # Build a 16384x32768 bf16 tensor of ones (~1 GiB) and move it to the XLA device.
    # 16384 * 32768 * 2 bytes (bfloat16) = 1,073,741,824 bytes = 1 GiB exactly.
    x_cpu = torch.ones(16384, 32768, dtype=torch.bfloat16)
    x = x_cpu.to(device)

    # Replicate across the entire mesh (no sharding on either dimension).
    xs.mark_sharding(x, mesh, (None, None))

    # Add 1 to the replicated tensor via torch.compile with the TT backend.
    class AddOne(torch.nn.Module):
        def forward(self, t):
            return t + 1

    compiled = torch.compile(AddOne(), backend="tt")
    result = compiled(x)

    result_cpu = result.cpu()
    expected = x_cpu + 1

    assert torch.allclose(result_cpu, expected), (
        f"Mismatch! Max diff: {(result_cpu - expected).abs().max()}"
    )

    print("result shape:", result_cpu.shape)
    print("result dtype:", result_cpu.dtype)
    print("result[0, :4]:", result_cpu[0, :4].tolist())
    print("All values correct:", torch.all(result_cpu == 2.0).item())

    return result_cpu


if __name__ == "__main__":
    main()
