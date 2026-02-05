# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
Pytest configuration for dual BH quietbox multi-host tests.

Opts into the shared multi-host distributed runtime fixture.
"""

from pathlib import Path

import pytest


@pytest.fixture(scope="session", autouse=True)
def configure_dual_bh_quietbox_topology(setup_distributed_env):
    """Configure environment for dual BH quietbox topology."""
    setup_distributed_env(
        topology="dual_bh_quietbox", script_dir=Path(__file__).parent
    )
