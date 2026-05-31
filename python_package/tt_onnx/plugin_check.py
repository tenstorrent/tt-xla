# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Preflight checks for pjrt_plugin_tt.so before ONNX compile."""

from __future__ import annotations

import ctypes
import os
from pathlib import Path


def default_pjrt_plugin_so() -> Path:
    import pjrt_plugin_tt

    return pjrt_plugin_tt.get_library_path()


def verify_pjrt_plugin_loads(plugin_so: Path | None = None) -> None:
    """
    Fail fast if pjrt_plugin_tt.so cannot be dlopen'd (common after tt-mlir rebuild).

    Raises:
        RuntimeError: with rebuild instructions when the plugin is stale.
    """
    so_path = Path(plugin_so or default_pjrt_plugin_so()).resolve()
    if not so_path.is_file():
        raise RuntimeError(f"PJRT plugin not found: {so_path}")

    lib_dir = so_path.parent / "lib"
    prev_ld = os.environ.get("LD_LIBRARY_PATH", "")
    extra = [str(lib_dir)] if lib_dir.is_dir() else []
    ttmlir_install = (
        so_path.parents[2]
        / "third_party"
        / "tt-mlir"
        / "install"
        / "lib"
    )
    if ttmlir_install.is_dir():
        extra.append(str(ttmlir_install))
    if extra:
        os.environ["LD_LIBRARY_PATH"] = ":".join(extra + ([prev_ld] if prev_ld else []))

    try:
        ctypes.CDLL(str(so_path))
    except OSError as exc:
        msg = str(exc)
        if "undefined symbol" in msg or "symbol lookup error" in msg:
            raise RuntimeError(
                f"PJRT plugin failed to load ({so_path}): {msg}\n\n"
                "This usually means pjrt_plugin_tt.so is stale relative to "
                "libTTMLIRCompiler.so (tt-mlir was rebuilt but the PJRT plugin was not).\n"
                "Rebuild on the reservation host:\n"
                "  source venv/activate\n"
                "  cmake --build build --target TTPJRTTTDylib -j\"$(nproc)\"\n"
                "Then re-run compile_add.py."
            ) from exc
        raise RuntimeError(f"PJRT plugin failed to load ({so_path}): {msg}") from exc
