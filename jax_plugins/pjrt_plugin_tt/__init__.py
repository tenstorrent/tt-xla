# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import os
import jax
import jax._src.xla_bridge as xb


def initialize():
    import pjrt_plugin_tt

    pjrt_plugin_parent_dir = list(pjrt_plugin_tt.__path__)[0]
    plugin_path = os.path.join(pjrt_plugin_parent_dir, "pjrt_plugin_tt.so")

    if not os.path.exists(plugin_path):
        raise FileNotFoundError(
            f"ERROR: Native library {plugin_path} does not exist. "
            f"This most likely indicates an issue with how {__package__} "
            f"was built or installed."
        )

    xb.register_plugin(
        "tt",
        priority=500,
        library_path=plugin_path,
        options=None,
    )

    jax.config.update("jax_platforms", "tt")
