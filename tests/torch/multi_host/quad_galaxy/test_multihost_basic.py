# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
Multi-host distributed tests for quad galaxy hardware.

All environment variables are auto-configured by conftest.py (including hosts and MPI agent).

Example:
    pytest -svv tests/torch/multi_host/quad_galaxy/test_multihost.py
"""

import os

import numpy as np
import pytest
import torch
import torch_xla
import torch_xla.distributed.spmd as xs
import torch_xla.runtime as xr
from infra.evaluators import ComparisonConfig, PccConfig, TorchComparisonEvaluator
from torch_xla.distributed.spmd import Mesh


def setup_spmd():
    """Helper to enable SPMD mode with Shardy conversion."""
    os.environ["CONVERT_SHLO_TO_SHARDY"] = "1"
    xr.use_spmd()


def create_device_mesh(mesh_shape) -> Mesh:
    """Helper to create a device mesh with specified shape."""
    num_devices = xr.global_runtime_device_count()
    device_ids = np.array(range(num_devices))
    mesh = Mesh(device_ids, mesh_shape, ("batch", "model"))
    return mesh


def test_simple_distributed_addition():
    """
    Verifies basic distributed tensor addition across multiple hosts.
    Creates two sharded tensors, adds them, and validates correctness.
    """

    class DistributedAdd(torch.nn.Module):
        def forward(self, a, b):
            return a + b

    xr.set_device_type("TT")
    setup_spmd()
    device = torch_xla.device()

    # Create mesh spanning all devices
    mesh = create_device_mesh((8, 16))

    # Create test tensors
    a_cpu = torch.ones(32, 32, dtype=torch.float32)
    b_cpu = torch.arange(1024, dtype=torch.float32).reshape(32, 32)
    expected_output = a_cpu + b_cpu

    # Move to device and shard
    a = a_cpu.to(device)
    b = b_cpu.to(device)
    xs.mark_sharding(a, mesh, (None, "model"))
    xs.mark_sharding(b, mesh, (None, "model"))

    # Compile and execute
    compiled_model = torch.compile(DistributedAdd(), backend="tt")
    compiled_model = compiled_model.to(device)
    output = compiled_model(a, b)

    # Validate correctness
    comparison_config = ComparisonConfig(
        pcc=PccConfig(required_pcc=0.9999), assert_on_failure=True
    )
    comparator = TorchComparisonEvaluator(comparison_config)
    comparator.evaluate(output.cpu(), expected_output)
