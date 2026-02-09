# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
Unified multi-host distributed tests that work across different topologies.

Tests explicitly specify supported topologies via @pytest.mark.parametrize.
Mesh shapes are automatically determined based on device count.

Example:
    # Run all tests on dual_bh_quietbox
    pytest -svv tests/torch/multi_host/experimental/test_multihost_basic.py -k "dual_bh_quietbox"

    # Run all tests on quad_galaxy
    pytest -svv tests/torch/multi_host/experimental/test_multihost_basic.py -k "quad_galaxy"

    # Run specific test on all topologies
    pytest -svv tests/torch/multi_host/experimental/test_multihost_basic.py::test_simple_distributed_addition
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


@pytest.mark.parametrize("topology", ["dual_bh_quietbox", "quad_galaxy", "dual_galaxy"])
def test_simple_distributed_addition(topology, configure_topology, mesh_shape):
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

    # Use mesh_shape from fixture (topology-aware)
    mesh = create_device_mesh(mesh_shape)

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


@pytest.mark.parametrize("topology", ["dual_bh_quietbox", "quad_galaxy", "dual_galaxy"])
def test_matmul_contracting_dim_sharded(topology, configure_topology, mesh_shape):
    """
    Matmul A @ B with contracting dimension (K) sharded across model axis.
    Each device holds a slice of A on K and B on K; local matmuls give partial
    sums; an all-reduce is required to form the full result. Validates correctness.
    """

    class MatmulContractingSharded(torch.nn.Module):
        def forward(self, a, b):
            return torch.matmul(a, b)

    xr.set_device_type("TT")
    setup_spmd()
    device = torch_xla.device()
    mesh = create_device_mesh(mesh_shape)

    batch_dim, model_dim = mesh_shape

    # Scale batch and K based on topology size
    if model_dim == 4:  # dual_bh_quietbox
        batch, M, K, N = 2, 16, 32, 16
    else:  # quad_galaxy or larger
        batch, M, K, N = 4, 16, 32, 16

    assert K % model_dim == 0, f"K={K} must be divisible by model_dim={model_dim}"

    A_cpu = torch.randn(batch, M, K, dtype=torch.float32)
    B_cpu = torch.randn(K, N, dtype=torch.float32)
    expected = torch.matmul(A_cpu, B_cpu)

    A = A_cpu.to(device)
    B = B_cpu.to(device)
    # A [batch, M, K]: shard on K (contracting dim) -> (None, None, "model")
    # B [K, N]: shard on K (first dim) -> ("model", None)
    xs.mark_sharding(A, mesh, (None, None, "model"))
    xs.mark_sharding(B, mesh, ("model", None))

    compiled = torch.compile(MatmulContractingSharded(), backend="tt")
    compiled = compiled.to(device)
    output = compiled(A, B)

    comparison_config = ComparisonConfig(
        pcc=PccConfig(required_pcc=0.9999), assert_on_failure=True
    )
    comparator = TorchComparisonEvaluator(comparison_config)
    comparator.evaluate(output.cpu(), expected)


@pytest.mark.parametrize("topology", ["dual_bh_quietbox", "quad_galaxy", "dual_galaxy"])
def test_matmul_batch_sharded(topology, configure_topology, mesh_shape):
    """
    Matmul A @ B with A sharded on batch. Each device holds a batch slice,
    B is replicated; result is sharded on batch. No all-reduce on result.
    """

    class MatmulBatchSharded(torch.nn.Module):
        def forward(self, a, b):
            return torch.matmul(a, b)

    xr.set_device_type("TT")
    setup_spmd()
    device = torch_xla.device()
    mesh = create_device_mesh(mesh_shape)

    batch_dim, model_dim = mesh_shape

    # Use batch_dim for the batch size to ensure even sharding
    batch, M, K, N = batch_dim, 16, 32, 16

    A_cpu = torch.randn(batch, M, K, dtype=torch.float32)
    B_cpu = torch.randn(K, N, dtype=torch.float32)
    expected = torch.matmul(A_cpu, B_cpu)

    A = A_cpu.to(device)
    B = B_cpu.to(device)
    # A sharded on batch; B replicated
    xs.mark_sharding(A, mesh, ("batch", None, None))
    xs.mark_sharding(B, mesh, (None, None))

    compiled = torch.compile(MatmulBatchSharded(), backend="tt")
    compiled = compiled.to(device)
    output = compiled(A, B)

    comparison_config = ComparisonConfig(
        pcc=PccConfig(required_pcc=0.9999), assert_on_failure=True
    )
    comparator = TorchComparisonEvaluator(comparison_config)
    comparator.evaluate(output.cpu(), expected)
