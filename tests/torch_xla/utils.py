# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
import os
import torch
import torch_xla.runtime as xr
from torch_xla.experimental import plugins
from torch_xla.distributed.spmd import Mesh
import numpy as np
from typing import Tuple


def setup_xla_environment_for_tp():
    # Basic XLA configuration
    num_devices = xr.global_runtime_device_count()
    os.environ["PJRT_DEVICE"] = "TT"
    os.environ["XLA_STABLEHLO_COMPILE"] = "1"
    os.environ["ENABLE_AUTO_PARALLEL"] = "TRUE"
    os.environ["CONVERT_SHLO_TO_SHARDY"] = "1"

    class TTPjrtPlugin(plugins.DevicePlugin):
        def library_path(self):
            return os.path.join(
                os.path.dirname(__file__), "../../../../build/src/tt/pjrt_plugin_tt.so"
            )

    plugins.register_plugin("TT", TTPjrtPlugin())

    xr.use_spmd()


def create_device_mesh(
    mesh_shape: Tuple[int, ...], mesh_names: Tuple[str, ...]
) -> Mesh:
    assert len(mesh_shape) == len(
        mesh_names
    ), "Mesh shape and names must match in length"
    num_devices = xr.global_runtime_device_count()
    assert (
        np.prod(mesh_shape) == num_devices
    ), "Mesh shape must match the number of devices"
    device_ids = np.array(range(num_devices))
    mesh = Mesh(device_ids, mesh_shape, mesh_names)
    os.environ["MESH_SHAPE"] = ",".join(map(str, mesh_shape))
    return mesh


def calculate_pcc(x: torch.Tensor, y: torch.Tensor) -> float:
    """Compute Pearson Correlation Coefficient."""
    assert x.shape == y.shape, "Input tensors must have the same shape"
    x_flat, y_flat = x.flatten(), y.flatten()
    x_flat, y_flat = x_flat.float(), y_flat.float()
    vx, vy = x_flat - x_flat.mean(), y_flat - y_flat.mean()
    denom = vx.norm() * vy.norm()
    if denom == 0:
        return float("nan")
    return float((vx @ vy) / denom)
