# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import torch
import torch_xla.core.xla_model as xm
import torch_xla.runtime as xr
from torch_xla.distributed.spmd import Mesh
import torch_xla.distributed.spmd as xs
import numpy as np
import os


def create_device_mesh():
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


def setup_spmd():
    print("Setting up XLA environment...")
    num_devices = xr.global_runtime_device_count()

    # Basic XLA configuration
    os.environ[
        "ENABLE_AUTO_PARALLEL"
    ] = "TRUE"  # Enables the auto parallel pass in tt-mlir
    os.environ[
        "CONVERT_SHLO_TO_SHARDY"
    ] = "1"  # Converts the StableHLO emitted by torch-xla to the Shardy dialect
    os.environ[
        "MESH_SHAPE"
    ] = f"1,{num_devices}"  # Sets the mesh shape used by the auto parallel pass

    # Initialize SPMD
    xr.use_spmd()
    print("XLA environment configured.")


def spmd_matmul_test():
    setup_spmd()

    # Connect to the XLA device
    device = xm.xla_device()

    # Create a device mesh
    mesh = create_device_mesh()

    # Define input tensors
    input_tensor = torch.randn(8, 16).to(device)
    weight_tensor = torch.randn(16, 32).to(device)
    intermediate_weight_tensor = torch.randn(32, 32).to(device)

    # Mark sharding on the input tensor
    print("Marking sharding on input_tensor", flush=True)
    xs.mark_sharding(input_tensor, mesh, ("model", None))

    print("Sharding marked on input_tensor", flush=True)

    # Perform matrix multiplication
    print("Performing matmul", flush=True)
    intermediate_tensor = torch.matmul(input_tensor, weight_tensor)

    output_tensor = torch.matmul(intermediate_tensor, intermediate_weight_tensor)

    print("Matmul completed", flush=True)

    # Print the output tensor
    print("Output tensor:", intermediate_tensor)


if __name__ == "__main__":
    # By default torch_xla uses the CPU device so we have to set it to TT device.
    xr.set_device_type("TT")

    spmd_matmul_test()
