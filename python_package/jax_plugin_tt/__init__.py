# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import os
from pathlib import Path

import jax._src.xla_bridge as xb

from pjrt_plugin_tt import setup_tt_metal_home, get_library_path
from .monkeypatch import _setup_monkey_patches


def initialize():
    setup_tt_metal_home()
    library_path = get_library_path()

    xb.register_plugin(
        "tt",
        priority=500,
        library_path=str(library_path),
        options=None,
    )

    _setup_monkey_patches()
