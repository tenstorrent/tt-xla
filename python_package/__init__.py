# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import os
from pathlib import Path

import jax._src.xla_bridge as xb

TT_PJRT_PLUGIN_NAME = "pjrt_plugin_tt.so"


def initialize():
    # Register bundled PJRT plugin.
    plugin_dir = Path(__file__).resolve().parent
    plugin_path = plugin_dir / "pjrt_plugin_tt.so"

    if not os.path.exists(plugin_path):
        raise FileNotFoundError(
            f"ERROR: Native library {plugin_path} does not exist. "
            f"This most likely indicates an issue with how {__package__} "
            f"was built or installed."
        )

    xb.register_plugin(
        "tt",
        priority=500,
        library_path=str(plugin_path),
        options=None,
    )

    # Export path to metal so it is accessible by bundled tt-metal installation.
    tt_metal_path = plugin_dir / "tt-mlir/install/tt-metal"
    os.environ["TT_METAL_HOME"] = str(tt_metal_path)
