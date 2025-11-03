# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
TT XLA Debug package.

This package provides Python bindings for the TT runtime debug hooks.
"""

import importlib.util
import sys
from pathlib import Path

TT_XLA_DEBUG_NAME = "tt_xla_debug.so"


def get_library_path() -> Path:
    """
    Get the path to the TT XLA debug library.
    """
    plugin_dir = Path(__file__).resolve().parent
    library_path = plugin_dir / TT_XLA_DEBUG_NAME

    if not library_path.exists():
        raise FileNotFoundError(
            f"ERROR: Native library {library_path} does not exist. "
            f"This most likely indicates an issue with how {__package__} "
            f"was built or installed."
        )

    return library_path


# Load the native module from the .so file in the same directory
_so_path = get_library_path()
spec = importlib.util.spec_from_file_location("tt_xla_debug", _so_path)

if spec is None or spec.loader is None:
    raise ImportError(f"Could not load module from {_so_path}")

_native_module = importlib.util.module_from_spec(spec)
sys.modules["tt_xla_debug._native"] = _native_module
spec.loader.exec_module(_native_module)

# Re-export all public symbols from the native module
__all__ = []
for name in dir(_native_module):
    if not name.startswith("_"):
        __all__.append(name)
        globals()[name] = getattr(_native_module, name)
