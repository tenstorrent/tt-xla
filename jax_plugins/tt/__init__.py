# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
import os

import jax
import jax._src.xla_bridge as xb


def initialize():
    print("Registering TT plugin with jax...")

    plugin_path = os.path.join(os.getcwd(), "build/src/tt/pjrt_plugin_tt.so")

    if not os.path.exists(plugin_path):
        print(
            f"WARNING: Native library {plugin_path} does not exist. "
            f"This most likely indicates an issue with how {__package__} "
            f"was built or installed."
        )

    xb.register_plugin(
        "tt",
        priority=500,
        library_path=plugin_path,
        options=None,
    )

    jax.config.update("jax_platforms", "tt,cpu")
