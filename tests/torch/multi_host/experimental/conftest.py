# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
Pytest configuration for unified multi-host tests with explicit topology parameterization.

Tests explicitly specify which topologies they support using pytest.mark.parametrize.
The mesh shape is automatically determined based on the device count.
"""

from pathlib import Path

import pytest
import torch_xla.runtime as xr

# Import the topology configurations
from tests.torch.multi_host.conftest import TOPOLOGIES


@pytest.fixture(scope="function")
def configure_topology(topology, setup_distributed_env):
    """
    Configure environment for the specified topology.

    This fixture sets up all necessary environment variables for distributed execution
    based on the topology parameter provided by test parameterization.

    Tests should include both 'topology' and 'configure_topology' in their parameters.

    Example:
        @pytest.mark.parametrize("topology", ["dual_bh_quietbox", "quad_galaxy"])
        def test_foo(topology, configure_topology, mesh_shape):
            # configure_topology will be called with the parameterized topology
            ...

    Args:
        topology: Topology name from test's @pytest.mark.parametrize("topology", [...])
        setup_distributed_env: Parent fixture from multi_host/conftest.py

    Returns:
        MultihostConfiguration object for the selected topology
    """
    script_dir = Path(__file__).parent
    return setup_distributed_env(topology=topology, script_dir=script_dir)


def get_mesh_shape_for_device_count(num_devices: int) -> tuple[int, int]:
    """
    Determine mesh shape based on total device count.

    Args:
        num_devices: Total number of devices across all hosts

    Returns:
        Tuple of (batch_dim, model_dim) for mesh shape

    Examples:
        8 devices -> (2, 4)   # dual_bh_quietbox
        16 devices -> (2, 8)  # not supported yet
        32 devices -> (4, 8)  # single_galaxy
        64 devices -> (8, 8)  # dual_galaxy
        128 devices -> (8, 16) # quad_galaxy
    """
    if num_devices == 8:
        return (2, 4)
    elif num_devices == 16:
        return (2, 8)
    elif num_devices == 32:
        return (4, 8)
    elif num_devices == 64:
        return (8, 8)
    elif num_devices == 128:
        return (8, 16)
    else:
        raise ValueError(
            f"Unsupported device count: {num_devices}. "
            f"Supported counts: 8, 16, 32, 64, 128"
        )


@pytest.fixture(scope="function")
def mesh_shape():
    """
    Fixture that provides the appropriate mesh shape for the current topology.

    Queries the actual device count and returns the corresponding mesh shape.
    This allows tests to be topology-agnostic.

    Returns:
        Tuple of (batch_dim, model_dim) for mesh construction
    """
    num_devices = xr.global_runtime_device_count()
    return get_mesh_shape_for_device_count(num_devices)
