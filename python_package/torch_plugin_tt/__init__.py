# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

from logging import Logger
import os

from torch_xla.experimental.plugins import DevicePlugin
from pjrt_plugin_tt import get_library_path, setup_tt_metal_home

import torch
import tt_torch  # registers "tt" backend for torch.compile

import torch_xla


class TTPlugin(DevicePlugin):
    """
    PyTorch/XLA plugin for Tenstorrent hardware.

    This plugin enables PyTorch/XLA to use Tenstorrent hardware by providing
    the path to the TT PJRT plugin binary and setting up the required environment.
    This uses the shared pjrt_plugin_tt package to avoid duplicating binaries.
    """

    def __init__(self):
        super().__init__()
        setup_tt_metal_home()

        # For using the PJRT plugin with `torch_xla` we need to set
        # `XLA_STABLEHLO_COMPILE` env variable to `1` to enable stablehlo compilation.
        # NOTE: This variable should soon be on by-default in `torch_xla`, but for now we need it.
        os.environ["XLA_STABLEHLO_COMPILE"] = "1"
        # HLO Debug and IR Debug flags are required for TorchXLA to attach useful location information to IR.
        # We rely on this for Codegen exporting.
        os.environ["XLA_HLO_DEBUG"] = "1"
        os.environ["XLA_IR_DEBUG"] = "1"
        print(
            f"WARNING: TT plugin is setting XLA_STABLEHLO_COMPILE to 1. This is required for TT PJRT plugin to work correctly."
        )
        torch_xla._XLAC._set_xla_all_numbers_special_scalars(True)

    def library_path(self) -> str:
        """Return the path to the TT PJRT plugin binary."""
        return str(get_library_path())


# This tells the torch dynamo to keep the models parameters bound to
# the GraphModule's state_dict, rather than consider them all as graph
# inputs. This allows the "tt" torch.compile backend to determine which
# inputs are parameters and which are user inputs.
torch._dynamo.config.inline_inbuilt_nn_modules = False
