# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
Pytest configuration for unified multi-host tests with explicit topology parameterization.

Tests explicitly specify which topologies they support using pytest.mark.parametrize.
The mesh shape is automatically determined based on the device count.

NOTE: TT_DISTRIBUTED_* (and related) environment variables are set by CI / the test
harness (e.g. eval $(python scripts/multihost_topology.py ...)) before pytest runs,
not by pytest fixtures during collection.
"""

import pytest


def get_mesh_shape_for_device_count(num_devices: int) -> tuple[int, int]:
    """
    Determine mesh shape based on total device count.

    Args:
        num_devices: Total number of devices across all hosts

    Returns:
        Tuple of (batch_dim, model_dim) for mesh shape

    Examples:
        8 devices -> (2, 4)   # dual_bh_quietbox
        16 devices -> (1, 16)  # dual t3k - can only be opened in 1x16 due to connectivity constraints
        32 devices -> (4, 8)  # single_galaxy
        64 devices -> (8, 8)  # dual_galaxy
        128 devices -> (8, 16) # quad_galaxy
    """
    if num_devices == 8:
        return (2, 4)
    elif num_devices == 16:
        return (1, 16)
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
def mesh_shape(topology):
    """
    Fixture that provides the appropriate mesh shape for the current topology.

    Derives mesh shape from the topology parameter to avoid querying devices
    during pytest collection (which would initialize the runtime too early).

    Args:
        topology: Topology name from test's @pytest.mark.parametrize("topology", [...])

    Returns:
        Tuple of (batch_dim, model_dim) for mesh construction
    """
    # Map topology to device count without querying hardware
    topology_device_counts = {
        "dual_bh_quietbox": 8,
        "dual_bh_loudbox_1x16": 16,
        "dual_t3k": 16,
        "single_galaxy": 32,
        "dual_galaxy": 64,
        "quad_galaxy": 128,
    }

    num_devices = topology_device_counts.get(topology)
    if num_devices is None:
        raise ValueError(
            f"Unknown topology '{topology}'. Known topologies: {list(topology_device_counts.keys())}"
        )

    return get_mesh_shape_for_device_count(num_devices)
