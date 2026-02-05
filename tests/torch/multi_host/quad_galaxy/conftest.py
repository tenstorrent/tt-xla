# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
Pytest configuration for quad galaxy multi-host tests.

Opts into the shared multi-host distributed runtime fixture.
"""

from pathlib import Path

import pytest


@pytest.fixture(scope="session", autouse=True)
def configure_quad_galaxy_topology(setup_distributed_env):
    """Configure environment for quad galaxy topology."""
    setup_distributed_env(
        topology="quad_galaxy", script_dir=Path(__file__).parent
    )
