# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import os
import sys

import torch
from pjrt_plugin_tt import (
    get_library_path,
    setup_tt_metal_home,
    setup_tt_pjrt_plugin_dir,
)
from ttxla_tools.logging import logger

# When torch_xla loads this module as a plugin entry point during its own
# __init__.py execution (via register_installed_plugins), torch_xla.experimental.plugins
# is already in sys.modules (imported at line 269, before register_installed_plugins
# at line 275). Access it directly to avoid a circular import where importing
# torch_xla would re-trigger register_installed_plugins before TTPlugin is defined.
if "torch_xla.experimental.plugins" in sys.modules:
    DevicePlugin = sys.modules["torch_xla.experimental.plugins"].DevicePlugin
else:
    from torch_xla.experimental.plugins import DevicePlugin


class TTPlugin(DevicePlugin):
    """
    PyTorch/XLA plugin for Tenstorrent hardware.

    This plugin enables PyTorch/XLA to use Tenstorrent hardware by providing
    the path to the TT PJRT plugin binary and setting up the required environment.
    This uses the shared pjrt_plugin_tt package to avoid duplicating binaries.
    """

    def __init__(self):
        super().__init__()
        setup_tt_pjrt_plugin_dir()
        setup_tt_metal_home()

        # For using the PJRT plugin with `torch_xla` we need to set
        # `XLA_STABLEHLO_COMPILE` env variable to `1` to enable stablehlo compilation.
        # NOTE: This variable should soon be on by-default in `torch_xla`, but for now we need it.
        os.environ["XLA_STABLEHLO_COMPILE"] = "1"

        logger.warning(
            "TT plugin is setting XLA_STABLEHLO_COMPILE to 1. This is required for TT PJRT plugin to work correctly."
        )
        import torch_xla as _torch_xla  # noqa: PLC0415 - lazy import avoids circular dependency
        import tt_torch  # noqa: F401, PLC0415 - registers "tt" backend for torch.compile

        _torch_xla._XLAC._set_xla_all_numbers_special_scalars(True)

    def library_path(self) -> str:
        """Return the path to the TT PJRT plugin binary."""
        return str(get_library_path())


# This tells the torch dynamo to keep the models parameters bound to
# the GraphModule's state_dict, rather than consider them all as graph
# inputs. This allows the "tt" torch.compile backend to determine which
# inputs are parameters and which are user inputs.
torch._dynamo.config.inline_inbuilt_nn_modules = False
