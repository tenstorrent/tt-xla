# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
Shared utilities and fixtures for multi-host distributed tests.
"""

import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict

import pytest

logger = logging.getLogger(__name__)


@dataclass
class MultihostConfiguration:
    """
    Configuration for a specific distributed hardware topology.

    This dataclass encapsulates all the environment variables needed to configure
    the distributed runtime for a specific multi-host hardware setup. Each field
    maps to a corresponding TT_DISTRIBUTED_* environment variable.

    Attributes:
        rank_binding: The rank binding configuration name that maps to a YAML file
            containing the rank-to-device mapping for this topology. This is used
            by the MPI runtime to assign devices to processes across hosts.
            Maps to: TT_DISTRIBUTED_RANK_BINDING
            Examples: "quad_galaxy", "2x4_double_bhqbae"

        controller_host_name: The hostname of the controller/coordinator node that
            orchestrates the distributed execution. This is typically the first host
            in the cluster where the main process runs.
            Maps to: TT_DISTRIBUTED_CONTROLLER_HOST_NAME
            Examples: "g05glx01", "forge-bh-01"

        hosts_list: Comma-separated list of all host machines participating in the
            distributed execution. The order should match the rank binding configuration.
            Maps to: TT_DISTRIBUTED_HOSTS_LIST
            Examples: "g05glx01,g05glx02,g05glx03,g05glx04", "forge-bh-01,forge-bh-02"

        remote_script_name: Name of the shell script used by MPI as the plm_rsh_agent
            to execute commands on remote hosts. This script handles SSH and docker
            exec commands to launch worker processes on each host.

            Only necessary for container to container tests.

            Maps to: TT_DISTRIBUTED_PLM_RSH_AGENT (via script_dir / remote_script_name)
            Default: "remote_docker.sh"

        btl_tcp_if_include: The network interface name to use for MPI byte transport
            layer (BTL) TCP communication between hosts. This controls which network
            interface MPI uses for inter-host data transfer.
            Maps to: TT_DISTRIBUTED_BTL_TCP_IF_INCLUDE

            "cnx1" for aus galaxies
            "enp10s0f1np1" for quietboxes
    """

    rank_binding: str
    controller_host_name: str
    hosts_list: str
    btl_tcp_if_include: str
    remote_script_name: str = "remote_docker.sh"


# Predefined topology configurations
TOPOLOGIES: Dict[str, MultihostConfiguration] = {
    "quad_galaxy": MultihostConfiguration(
        rank_binding="quad_galaxy",
        controller_host_name="g05glx01",
        hosts_list="g05glx01,g05glx02,g05glx03,g05glx04",
        btl_tcp_if_include="cnx1",
    ),
    "dual_bh_quietbox": MultihostConfiguration(
        rank_binding="2x4_double_bhqbae",
        controller_host_name="forge-qbae-01",
        hosts_list="forge-qbae-01,forge-qbae-02",
        btl_tcp_if_include="enp10s0f1np1",
        remote_script_name=None,  # not used for dual BH quietbox since this runs on baremetal
    ),
}


def get_distributed_worker_path():
    """
    Get path to distributed worker binary.

    First checks TT_DISTRIBUTED_WORKER_PATH environment variable.
    If not set, constructs path from TT_PJRT_PLUGIN_DIR or TTMLIR_TOOLCHAIN_DIR.

    Returns:
        str: Path to the distributed worker binary

    Raises:
        AssertionError: If worker path is not set and cannot be constructed, or if path doesn't exist
    """
    # Check if explicitly set
    worker_path = os.environ.get("TT_DISTRIBUTED_WORKER_PATH")
    if worker_path:
        assert os.path.exists(
            worker_path
        ), f"Distributed worker file does not exist at path: {worker_path}"
        return worker_path

    # Try TT_PJRT_PLUGIN_DIR first
    pjrt_plugin_dir = os.environ.get("TT_PJRT_PLUGIN_DIR")
    if pjrt_plugin_dir:
        worker_path = os.path.join(
            pjrt_plugin_dir, "bin/ttmlir/runtime/distributed/worker"
        )
        assert os.path.exists(
            worker_path
        ), f"Distributed worker file does not exist at path: {worker_path}"
        return worker_path

    # Fallback to TTMLIR_TOOLCHAIN_DIR
    toolchain_dir = os.environ.get("TTMLIR_TOOLCHAIN_DIR")
    assert toolchain_dir, (
        "Neither TT_DISTRIBUTED_WORKER_PATH, TT_PJRT_PLUGIN_DIR, nor TTMLIR_TOOLCHAIN_DIR "
        "environment variable is set"
    )

    worker_path = str(
        Path(toolchain_dir) / "bin" / "ttmlir" / "runtime" / "distributed" / "worker"
    )
    assert os.path.exists(
        worker_path
    ), f"Distributed worker file does not exist at path: {worker_path}"

    return worker_path


@pytest.fixture(scope="session")
def setup_distributed_env():
    """
    Session-scoped fixture to configure distributed runtime environment variables.

    This fixture is NOT autouse - test directories must explicitly opt-in by calling it.

    Usage in test directory conftest.py:
        @pytest.fixture(scope="session", autouse=True)
        def configure_topology(setup_distributed_env):
            return setup_distributed_env(topology="quad_galaxy", script_dir=Path(__file__).parent)

    Args:
        topology: Name of the topology from TOPOLOGIES dict
        script_dir: Directory containing the remote_docker.sh script
    """
    original_values = {}

    def _setup(topology: str, script_dir: Path):
        """Configure environment for specified topology."""
        if topology not in TOPOLOGIES:
            raise ValueError(
                f"Unknown topology '{topology}'. Available: {list(TOPOLOGIES.keys())}"
            )

        topo = TOPOLOGIES[topology]
        remote_script = str(script_dir / topo.remote_script_name)

        # Variables to set
        env_vars = {
            "TT_RUNTIME_ENABLE_DISTRIBUTED": "1",
            "TT_DISTRIBUTED_WORKER_PATH": get_distributed_worker_path(),
            "TT_DISTRIBUTED_RANK_BINDING": topo.rank_binding,
            "TT_DISTRIBUTED_CONTROLLER_HOST_NAME": topo.controller_host_name,
            "TT_DISTRIBUTED_BTL_TCP_IF_INCLUDE": topo.btl_tcp_if_include,
            "TT_DISTRIBUTED_HOSTS_LIST": topo.hosts_list,
            "TT_DISTRIBUTED_PLM_RSH_AGENT": remote_script,
        }

        # Set environment variables
        for key, value in env_vars.items():
            original_values[key] = os.environ.get(key)
            os.environ[key] = value

        logger.info("setup_distributed_env: topology=%s, env: %s", topology, env_vars)

        return topo

    yield _setup

    # Cleanup: restore original values or unset
    for key, original_value in original_values.items():
        if original_value is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = original_value
