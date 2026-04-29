# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
from .logger import tt_init_logger

logger = tt_init_logger(__name__)


def determine_mesh_shape(num_devices: int, use_2d_mesh: bool) -> tuple[int, int]:
    if use_2d_mesh:
        # Use predefined mesh shapes based on number of devices
        mesh_shapes = {
            2: (1, 2),
            4: (2, 2),
            8: (2, 4),
            16: (4, 4),
            32: (4, 8),
        }
        if num_devices in mesh_shapes:
            mesh_shape = mesh_shapes[num_devices]
            logger.info(
                f"Using predefined mesh shape for {num_devices} devices: {mesh_shape}"
            )
            return mesh_shape
        else:
            # Fallback to computation for unsupported device counts
            logger.warning(
                f"Using fallback computation for {num_devices} devices (not in predefined shapes)"
            )
            mesh_dim1 = int(num_devices**0.5)
            while num_devices % mesh_dim1 != 0:
                mesh_dim1 -= 1
            mesh_dim2 = num_devices // mesh_dim1
            mesh_shape = (mesh_dim1, mesh_dim2)
            logger.info(f"Computed mesh shape: {mesh_shape}")
            return mesh_shape
    else:
        # For 1D mesh, all devices are in one dimension.
        mesh_shape = (1, num_devices)
        logger.info(f"Using 1D mesh shape for {num_devices} devices: {mesh_shape}")
        return mesh_shape


def prev_power_of_2(n: int) -> int:
    """The previous power of 2 (inclusive)"""
    return 0 if n <= 0 else 1 << (n.bit_length() - 1)
