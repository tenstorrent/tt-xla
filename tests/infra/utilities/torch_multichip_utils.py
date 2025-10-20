# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
import os
from typing import Tuple

import numpy as np
import torch_xla.runtime as xr
from torch_xla.distributed.spmd import Mesh
# from torch_xla.experimental import plugins


def get_mesh(mesh_shape: Tuple[int], mesh_names: Tuple[str]) -> Mesh:
    """
    Creates and returns a Mesh object based on the provided mesh shape and mesh names.
    Args:
        mesh_shape (tuple or None): The shape of the mesh to create. If None, no mesh is created and None is returned.
        mesh_names (list or sequence): The names corresponding to each dimension of the mesh.
    Returns:
        Mesh or None: A Mesh object if mesh_shape is provided, otherwise None.
    """

    num_devices = xr.global_runtime_device_count()
    device_ids = np.array(range(num_devices))
    mesh = Mesh(device_ids, mesh_shape, mesh_names) if mesh_shape is not None else None
    print(f"Created device mesh: {mesh_shape} with {num_devices} devices.")
    return mesh


def enable_spmd():
    """
    Enable torch_xla SPMD mode.

    Note:
        - This cannot be disabled once set. See: https://github.com/pytorch/xla/issues/9578
    """
    # In the pytorch-xla fork this enables the ConvertStableHloToSdy pass.
    # The tt-mlir stablehlo compiler pipeline expects input shlo from pytorch/xla to contain shardy annotations.
    os.environ["CONVERT_SHLO_TO_SHARDY"] = "1"
    xr.set_device_type("TT")
    xr.use_spmd()


# # TODO this shouldn't be here, but a common utility
# def setup_xla_environment():
#     """Setup TensorTrent environment and plugin."""
#     os.environ["PJRT_DEVICE"] = "TT"
#     os.environ["XLA_STABLEHLO_COMPILE"] = "1"
#     os.environ["XLA_ALWAYS_ALLREDUCE"] = "1"
#     os.environ["CONVERT_SHLO_TO_SHARDY"] = "1"

#     class TTPjrtPlugin(plugins.DevicePlugin):
#         def library_path(self):
#             # Find tt-xla repo root by traversing up from current file
#             current_dir = os.path.dirname(os.path.abspath(__file__))
#             while current_dir != "/" and not os.path.exists(os.path.join(current_dir, "build")):
#                 current_dir = os.path.dirname(current_dir)
#             return os.path.join(current_dir, "build/src/tt/pjrt_plugin_tt.so")

#     plugins.register_plugin("TT", TTPjrtPlugin())
#     xr.set_device_type("TT")
#     xr.use_spmd()