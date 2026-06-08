# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
from .vllm_distributed_utils import ParallelismMode

from .logger import tt_init_logger

logger = tt_init_logger(__name__)


def determine_mesh_shape(
    num_devices: int, parallel_mode: ParallelismMode
) -> tuple[int, int]:
    if parallel_mode == ParallelismMode.DATA_PARALLEL_ONLY:
        mesh_shape = (num_devices, 1)
        logger.info(f"Using data-parallel mesh shape: {mesh_shape}")
        return mesh_shape

    if parallel_mode == ParallelismMode.TENSOR_PARALLEL_ONLY_1D:
        mesh_shape = (1, num_devices)
        logger.info(f"Using 1D tensor-parallel mesh shape: {mesh_shape}")
        return mesh_shape

    if parallel_mode in (
        ParallelismMode.TENSOR_PARALLEL_ONLY_2D,
        ParallelismMode.DATA_TENSOR_PRALLEL,
    ):
        # Use predefined mesh shapes based on number of devices.
        # For DATA_TENSOR_PRALLEL the axes are (data_parallel, tensor_parallel).
        mesh_shapes = {
            2: (1, 2),
            4: (2, 2),
            8: (2, 4),
            16: (4, 4),
            32: (4, 8),
        }
        if parallel_mode == ParallelismMode.DATA_TENSOR_PRALLEL:
            # BH galaxy (32 chips): 8x4 = 8 data-parallel replicas x 4-way TP.
            # tp=4 divides Devstral's 8 KV heads (2/device); the default (4,8)
            # would be dp4/tp8.
            mesh_shapes[32] = (8, 4)
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

    if parallel_mode == ParallelismMode.DISABLED:
        mesh_shape = (1, 1)
        logger.info(f"Using disabled mesh shape: {mesh_shape}")
        return mesh_shape

    raise ValueError(f"Unsupported parallel mode: {parallel_mode}")


def prev_power_of_2(n: int) -> int:
    """The previous power of 2 (inclusive)"""
    return 0 if n <= 0 else 1 << (n.bit_length() - 1)
