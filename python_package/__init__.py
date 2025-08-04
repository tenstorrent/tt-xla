# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import os
from pathlib import Path

import jax._src.xla_bridge as xb

from .monkeypatch import apply_patches, get_monkeypatches

TT_PJRT_PLUGIN_NAME = "pjrt_plugin_tt.so"


def _register_plugin():
    """
    Register the Tenstorrent PJRT plugin with JAX.

    This function:
    - Locates the PJRT plugin shared library
    - Registers it with JAX's XLA bridge
    - Sets up the TT_METAL_HOME environment variable

    Raises:
        FileNotFoundError: If the PJRT plugin library is not found
    """
    plugin_dir = Path(__file__).resolve().parent
    plugin_path = plugin_dir / TT_PJRT_PLUGIN_NAME

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


def _setup_monkey_patches():
    """
    Set up and apply monkey patches for JAX and Flax compatibility.

    This function applies monkey patches to jax.nn.gelu for Tenstorrent optimization
    and flax.linen.Module.apply for weight marking.
    """
    # Get and apply monkey patches
    monkeypatches = get_monkeypatches()
    apply_patches(monkeypatches)


def initialize():
    """
    Initialize the Tenstorrent JAX plugin.

    This is the main entry point that should be called to set up the plugin.
    It performs the following operations:
    1. Registers the PJRT plugin with JAX
    2. Sets up monkey patches for framework compatibility

    This function should be called once before using JAX with Tenstorrent hardware.
    """
    _register_plugin()
    _setup_monkey_patches()
