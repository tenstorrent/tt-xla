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
        # Axes are (batch, model): "model" is the tensor-parallel axis (and for
        # DATA_TENSOR_PRALLEL "batch" is the data-parallel axis).
        #
        # BH galaxy (32 chips) is forced to 8x4 = (batch=8, model=4) for BOTH
        # TENSOR_PARALLEL_ONLY_2D and DATA_TENSOR_PRALLEL. The model (TP) axis = 4
        # divides Devstral's 8 KV heads evenly (2 heads/device), so SDPA-decode
        # uses ~60 cores/head and stays under tt-metal's 64-cores/head tree-
        # reduction cap. The natural (4,8) instead gives model=8 -> 1 head/device
        # -> 120 cores/head -> TT_FATAL in sdpa_decode_program_factory.
        mesh_shapes = {
            2: (1, 2),
            4: (2, 2),
            8: (2, 4),
            16: (4, 4),
            32: (8, 4),
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

    if parallel_mode == ParallelismMode.DISABLED:
        mesh_shape = (1, 1)
        logger.info(f"Using disabled mesh shape: {mesh_shape}")
        return mesh_shape

    raise ValueError(f"Unsupported parallel mode: {parallel_mode}")


def prev_power_of_2(n: int) -> int:
    """The previous power of 2 (inclusive)"""
    return 0 if n <= 0 else 1 << (n.bit_length() - 1)
