# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
import os
from typing import Tuple

import numpy as np
import torch_xla.runtime as xr
from torch_xla.distributed.spmd import Mesh


def get_mesh_shape_for_device_count(num_devices: int) -> Tuple[int, int]:
    """
    Map total device count to a 2D (batch, model) mesh shape for multi-host / multichip runs.

    Args:
        num_devices: Total devices visible to the PJRT runtime (all hosts).

    Returns:
        ``(batch_dim, model_dim)`` with product equal to ``num_devices``.

    Examples:
        8 -> (2, 4)    # e.g. dual_bh_quietbox
        16 -> (1, 16)  # e.g. dual T3K / loudbox 1x16
        32 -> (4, 8)   # e.g. single galaxy
        64 -> (8, 8)   # e.g. dual galaxy
        128 -> (8, 16) # e.g. quad galaxy
    """
    if num_devices == 8:
        return (2, 4)
    if num_devices == 16:
        return (1, 16)
    if num_devices == 32:
        return (4, 8)
    if num_devices == 64:
        return (8, 8)
    if num_devices == 128:
        return (8, 16)
    raise ValueError(
        f"Unsupported device count: {num_devices}. "
        f"Supported counts: 8, 16, 32, 64, 128"
    )


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
    xr.use_spmd()
