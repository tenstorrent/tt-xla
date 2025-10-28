# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
TT PJRT device package.

This package contains the actual PJRT plugin binary and tt-mlir dependencies.
Both JAX and PyTorch/XLA plugins reference this package.
"""

import os
from pathlib import Path

TT_PJRT_PLUGIN_NAME = "pjrt_plugin_tt.so"


def setup_tt_metal_home():
    """
    Setup the `TT_METAL_RUNTIME_ROOT` environment variable by looking for the `tt-metal` installation.
    If user already has set the `TT_METAL_RUNTIME_ROOT` environment variable, we will not override it - we
    will only verify that the path exists and raise an error if it does not.

    For setting the `tt-metal` home path we prioritize the path in the wheel package,
    if it does not exist, we use the path in the source tree.
    """
    plugin_dir = Path(__file__).resolve().parent
    tt_metal_path_in_whl = plugin_dir / "tt-metal"

    tt_xla_root = plugin_dir.parent.parent
    tt_metal_path_in_source = (
        tt_xla_root
        / "third_party"
        / "tt-mlir"
        / "src"
        / "tt-mlir"
        / "third_party"
        / "tt-metal"
        / "src"
        / "tt-metal"
    )

    # Check if path to `tt-metal` has already been set in the environment.
    # If it has verify that it exists, otherwise raise an error.
    # We will not override this environment variable if it is already set.
    user_override = os.getenv("TT_METAL_RUNTIME_ROOT")
    if user_override is not None:
        if Path(user_override).exists():
            print(f"Using TT-Metal path from environment variable: {user_override}")
            return

        raise FileNotFoundError(
            f"ERROR: TT-Metal installation not found at {user_override}. "
            f"This location was specified by the TT_METAL_RUNTIME_ROOT environment variable, "
            f"please check that the path is correct."
        )

    # We need to set the `TT_METAL_RUNTIME_ROOT` environment variable.
    # First priority is the path in the wheel package, if this doesn't exist - i.e. we are not installed via wheel,
    # then we use the path in the source tree.
    if tt_metal_path_in_whl.exists():
        os.environ["TT_METAL_RUNTIME_ROOT"] = str(tt_metal_path_in_whl)
        print(f"Using TT-Metal from wheel package: {tt_metal_path_in_whl}")
        return

    if tt_metal_path_in_source.exists():
        os.environ["TT_METAL_RUNTIME_ROOT"] = str(tt_metal_path_in_source)
        print(f"Using TT-Metal from the source tree: {tt_metal_path_in_source}")
        return

    raise FileNotFoundError(
        f"ERROR: TT-Metal installation could not be found."
        f"This most likely indicates an issue with how {__package__} "
        f"was built or installed."
    )


def get_library_path() -> Path:
    """
    Get the path to the TT PJRT plugin library.
    """
    plugin_dir = Path(__file__).resolve().parent
    library_path = plugin_dir / TT_PJRT_PLUGIN_NAME

    if not library_path.exists():
        raise FileNotFoundError(
            f"ERROR: Native library {library_path} does not exist. "
            f"This most likely indicates an issue with how {__package__} "
            f"was built or installed."
        )

    return library_path
