# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import os

import jax
import jax._src.xla_bridge as xb
import pytest
#from infra.device_connector import device_connector


def initialize():
    path = os.path.join(os.path.dirname(__file__), "../build/src/tt/pjrt_plugin_tt.so")
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Could not find tt_pjrt C API plugin at {path}, have you compiled the project?"
        )

    plugin = xb.register_plugin("tt", priority=500, library_path=path, options=None)
    jax.config.update("jax_platforms", "tt,cpu")


@pytest.fixture(scope="session", autouse=True)
def setup_session():
    # Added to prevent `PJRT_Api already exists for device type tt` error.
    # Will be removed completely soon.
    #if not device_connector.is_initialized():
    initialize()
