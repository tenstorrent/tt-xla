# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
import pytest
import torch
import torch_xla.core.xla_model as xm
import torch_xla.runtime as xr
import torch_xla
import numpy as np
from torch_xla.distributed.spmd import Mesh
import torch_xla.distributed.spmd as xs
import os


def setup_spmd():
    print("Setting up XLA environment...")
    num_devices = xr.global_runtime_device_count()

    # Basic XLA configuration
    os.environ["ENABLE_AUTO_PARALLEL"] = (
        "TRUE"  # Enables the auto parallel pass in tt-mlir
    )
    os.environ["CONVERT_SHLO_TO_SHARDY"] = (
        "1"  # Converts the StableHLO emitted by torch-xla to the Shardy dialect
    )
    os.environ["MESH_SHAPE"] = (
        f"1,{num_devices}"  # Sets the mesh shape used by the auto parallel pass
    )

    # Initialize SPMD
    xr.use_spmd()
    print("XLA environment configured.")


def create_device_mesh() -> Mesh:
    """
    Create device mesh for tensor parallelism.

    Args:
        num_devices: Total number of devices
        mesh_shape: Shape of the device mesh (batch_dim, model_dim)

    Returns:
        Mesh object for SPMD operations
    """
    num_devices = xr.global_runtime_device_count()
    mesh_shape = (1, num_devices)
    device_ids = np.array(range(num_devices))
    mesh = Mesh(device_ids, mesh_shape, ("batch", "model"))
    print(f"Created device mesh: {mesh_shape} with {num_devices} devices")
    return mesh


def test_multi_shard():
    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.factor = torch.eye(4)

        def forward(self, x):
            return x @ self.factor

    # By default torch_xla uses the CPU device so we have to set it to TT device.
    xr.set_device_type("TT")
    setup_spmd()
    mesh = create_device_mesh()
    device = xm.xla_device()

    model: Model = Model()

    # set up inputs and model
    x = torch.ones((4, 4), dtype=torch.bfloat16)
    n_loops = 1

    # Connect the device.

    # Move inputs and model to device.
    x = x.to(device)
    model.factor = model.factor.to(device)

    xs.mark_sharding(
        x,
        mesh,
        (
            None,
            "model",
        ),
    )
    xs.mark_sharding(model.factor, mesh, ("model", None))
    model.compile(backend="tt")

    # compile the model
    with torch.no_grad():
        for _ in range(n_loops):
            y = model(x)
            print(y)


"""
def test_inplace_add_multiloop():
    class AddModule(torch.nn.Module):
        def __init__(self):
            super().__init__()

        def forward(self, x):
            x = x + 1
            return x

    # By default torch_xla uses the CPU device so we have to set it to TT device.
    xr.set_device_type("TT")

    # set up inputs and model
    x = torch.zeros((3, 3), dtype=torch.bfloat16)

    model = AddModule()
    model.compile(backend="tt")

    output = None
    n_loops = 100

    # Connect the device.
    device = xm.xla_device()

    # Move inputs and model to device.
    x = x.to(device)
    model = model.to(device)

    # compile the model

    with torch.no_grad():
        for _ in range(n_loops):
            x = model(x)
            print(x)

    # result = x.to("cpu")
    # assert result.equal(torch.ones(3,3)*n_loops)
"""
