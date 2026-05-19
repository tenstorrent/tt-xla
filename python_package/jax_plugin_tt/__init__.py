# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import os
from pathlib import Path

import jax._src.xla_bridge as xb
from pjrt_plugin_tt import (
    get_library_path,
    setup_tt_metal_home,
    setup_tt_pjrt_plugin_dir,
)

from .monkeypatch import _patch_triton_for_non_gpu_host, setup_monkey_patches

# Install the triton null-driver shim at module-load time so that any
# subsequent `import easydel` works on hosts without a GPU. JAX only invokes
# `initialize()` lazily (on first device access), which can be after user
# code has already tried to import EasyDeL — too late to install patches
# from there. This is safe to run unconditionally: it's a no-op when
# triton is absent or when a real GPU driver is active.
_patch_triton_for_non_gpu_host()


def initialize():
    setup_tt_pjrt_plugin_dir()
    setup_tt_metal_home()
    library_path = get_library_path()

    xb.register_plugin(
        "tt",
        priority=500,
        library_path=str(library_path),
        options=None,
    )

    setup_monkey_patches()
