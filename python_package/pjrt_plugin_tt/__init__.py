# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
TT PJRT device package.

This package contains the actual PJRT plugin binary and tt-mlir dependencies.
Both JAX and PyTorch/XLA plugins reference this package.
"""

import atexit
import ctypes
import os
from pathlib import Path

from ttxla_tools.logging import logger

TT_PJRT_PLUGIN_NAME = "pjrt_plugin_tt.so"

_shutdown_hook_registered = False


def _device_bdfs_in_umd_order():
    """
    PCI BDFs of all Tenstorrent devices, sorted by BDF.

    This is the order UMD's `PCIDevice::enumerate_devices()` indexes with an
    integer token -- not `/dev/tenstorrent/<n>` node numbers. Empty when no
    devices are visible in sysfs.
    """
    dev_dir = Path("/dev/tenstorrent")
    if not dev_dir.exists():
        return []

    bdfs = []
    for node in dev_dir.iterdir():
        try:
            bdf = Path(
                f"/sys/class/tenstorrent/tenstorrent!{node.name}/device"
            ).resolve()
        except OSError:
            continue  # UMD skips devices it cannot read; do the same.
        if bdf.name:
            bdfs.append(bdf.name)

    return sorted(bdfs)


def normalize_tt_visible_devices():
    """
    WORKAROUND (tt-xla#5521) -- NOT a fix. Delete once tt-metal is fixed.

    Rewrites an integer `TT_VISIBLE_DEVICES` to the equivalent PCI BDFs, which
    are immune to the double-application that rejects every integer except 0.
    The fix is two lines in tt-metal's `Cluster::open_driver()` (pass the mock
    descriptor's own chips as `target_devices`); when that lands, delete this
    and its three call sites.

    Inert unless translation is both needed and safe: no-op when the variable is
    unset or empty, when any token is not a plain integer (so BDFs pass through
    and this is idempotent), when no devices are visible in sysfs, and when an
    index is out of range -- leaving UMD to report that itself.
    """
    value = os.getenv("TT_VISIBLE_DEVICES")
    if not value or not value.strip():
        return

    tokens = [token.strip() for token in value.split(",") if token.strip()]
    if not tokens or not all(token.isdigit() for token in tokens):
        return

    bdfs = _device_bdfs_in_umd_order()
    if any(int(token) >= len(bdfs) for token in tokens):
        return

    normalized = ",".join(bdfs[int(token)] for token in tokens)
    os.environ["TT_VISIBLE_DEVICES"] = normalized
    logger.info(
        f"Normalized TT_VISIBLE_DEVICES from '{value}' to '{normalized}' so device "
        f"selection survives UMD chip-id renumbering (tt-xla#5521)."
    )


def setup_tt_pjrt_plugin_dir():
    """
    Setup the `TT_PJRT_PLUGIN_DIR` environment variable by looking for the `pjrt_plugin_tt.so` file.
    If user already has set the `TT_PJRT_PLUGIN_DIR` environment variable, we will not override it - we
    will only verify that the path exists and raise an error if it does not.
    """
    user_override = os.getenv("TT_PJRT_PLUGIN_DIR")
    if user_override is not None:
        if Path(user_override).exists():
            logger.info(
                f"Using PJRT plugin directory from environment variable: {user_override}"
            )
            return
        raise FileNotFoundError(
            f"ERROR: PJRT plugin directory not found at {user_override}. "
            f"This location was specified by the TT_PJRT_PLUGIN_DIR environment variable, "
            f"please check that the path is correct."
        )

    plugin_dir = Path(__file__).resolve().parent
    if plugin_dir.exists():
        os.environ["TT_PJRT_PLUGIN_DIR"] = str(plugin_dir)
        logger.info(f"Using PJRT plugin directory: {plugin_dir}")
        return

    raise FileNotFoundError(
        f"ERROR: PJRT plugin directory could not be found. This most likely indicates an issue with how {__package__} "
        f"was built or installed."
    )


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
            logger.info(
                f"Using TT-Metal path from environment variable: {user_override}"
            )
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
        logger.info(f"Using TT-Metal from wheel package: {tt_metal_path_in_whl}")
        return

    if tt_metal_path_in_source.exists():
        os.environ["TT_METAL_RUNTIME_ROOT"] = str(tt_metal_path_in_source)
        logger.info(f"Using TT-Metal from the source tree: {tt_metal_path_in_source}")
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


def register_shutdown_hook() -> None:
    """
    Register a Python `atexit` handler that drives controlled shutdown of
    plugin-owned resources before the interpreter tears down modules and
    destroys the GIL. The handler invokes the exported `tt_pjrt_shutdown`
    C symbol via ctypes.

    Safe to call multiple times; the hook is registered at most once.
    """
    global _shutdown_hook_registered
    if _shutdown_hook_registered:
        return

    library_path = get_library_path()
    lib = ctypes.CDLL(str(library_path))
    lib.tt_pjrt_shutdown.restype = None
    lib.tt_pjrt_shutdown.argtypes = []

    atexit.register(lib.tt_pjrt_shutdown)
    _shutdown_hook_registered = True
