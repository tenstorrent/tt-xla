# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import os
from pathlib import Path

import jax._src.xla_bridge as xb

from pjrt_plugin_tt import setup_tt_metal_home, get_library_path
from .monkeypatch import apply_patches, get_monkeypatches


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
    # Register bundled PJRT plugin.
    setup_tt_metal_home()
    library_path = get_library_path()

    xb.register_plugin(
        "tt",
        priority=500,
        library_path=str(library_path),
        options=None,
    )

    _setup_monkey_patches()
