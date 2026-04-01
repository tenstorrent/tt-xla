# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
Shared utilities and fixtures for multi-host distributed tests.
"""

import os
from pathlib import Path

import pytest
from ttxla_tools.logging import logger

from tests.torch.multi_host.topology import (
    MultihostConfiguration,
    TOPOLOGIES,
    get_distributed_worker_path,
    get_env_vars,
)

__all__ = [
    "MultihostConfiguration",
    "TOPOLOGIES",
    "get_distributed_worker_path",
]


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
        env_vars = get_env_vars(topology, script_dir)

        for key, value in env_vars.items():
            original_values[key] = os.environ.get(key)
            os.environ[key] = value

        logger.info(
            "setup_distributed_env: topology={}\n{}",
            topology,
            "\n".join([f"{k}={v}" for k, v in env_vars.items()]),
        )

        return TOPOLOGIES[topology]

    yield _setup

    # Cleanup: restore original values or unset
    for key, original_value in original_values.items():
        if original_value is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = original_value
