# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
import os
import sys

import jax
import jax._src.xla_bridge as xb


def init_device():
    priority = 500
    backend = "tt"
    path = os.path.join(os.getcwd(), "/localdev/upantelic/tt-blacksmith/third_party/tt-xla/build/src/tt/pjrt_plugin_tt.so")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Could not find tt_pjrt C API plugin at {path}")

    print("Loading tt_pjrt C API plugin", file=sys.stderr)
    xb.discover_pjrt_plugins()

    plugin = xb.register_plugin(backend, priority=priority, library_path=path, options=None)
    print("Loaded", file=sys.stderr)
    jax.config.update("jax_platforms", "tt,cpu")
