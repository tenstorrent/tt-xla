# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
import torch_xla.runtime as xr
from torch_xla.distributed.spmd import Mesh
import numpy as np
import os


def get_mesh(mesh_shape, mesh_names):
    num_devices = xr.global_runtime_device_count()
    device_ids = np.array(range(num_devices))
    mesh = Mesh(device_ids, mesh_shape, mesh_names) if mesh_shape is not None else None
    return mesh


def enable_spmd():
    os.environ["CONVERT_SHLO_TO_SHARDY"] = "1"
    xr.use_spmd()
