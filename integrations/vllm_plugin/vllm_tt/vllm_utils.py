# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
from typing import Any

from .logger import tt_init_logger
from .vllm_distributed_utils import ParallelismMode

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
        ParallelismMode.DATA_TENSOR_PARALLEL,
    ):
        # Use predefined mesh shapes based on number of devices.
        # Axes are (batch, model): "model" is the tensor-parallel axis (and for
        # DATA_TENSOR_PRALLEL "batch" is the data-parallel axis).
        #
        # BH galaxy (32 chips): EXPERIMENT - forced to 4x8 = (batch=4, model=8).
        # The full Devstral-123B does not fit in a TP-4 weight slice (~32 GB/
        # device at 32 GB DRAM) even in bfp8, so it needs model=8 (1/8 per
        # device, ~16 GB). Caveat: model=8 -> Devstral's 8 KV heads become
        # 1/device -> ~120 cores/head -> may TT_FATAL in sdpa_decode (tree
        # reduction cap is 64 cores/head). 8x4 = (8,4) avoids that but only
        # fits smaller models (e.g. Qwen3-32B); see the qwen3-32b-bh-galaxy
        # branch. Trying 4x8 to see whether the 123B runs end-to-end.
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

    if parallel_mode == ParallelismMode.DISABLED:
        mesh_shape = (1, 1)
        logger.info(f"Using disabled mesh shape: {mesh_shape}")
        return mesh_shape

    raise ValueError(f"Unsupported parallel mode: {parallel_mode}")


def prev_power_of_2(n: int) -> int:
    """The previous power of 2 (inclusive)"""
    return 0 if n <= 0 else 1 << (n.bit_length() - 1)


def apply_hidden_layer_override(
    hf_config: Any,
    target_num_layers: int,
) -> tuple[int | None, int | None]:
    """Apply decoder hidden-layer override for root or text sub-config.

    Plain LM configs expose num_hidden_layers attribute on the root config,
    while multimodal model configs uses text_config for num_hidden_layers.
    This helper resolves the right config object and applies the override only
    when 0 < target_num_layers < original_num_layers.

    Returns:
        A tuple (original_num_layers, target_num_layers) when an override
        is applied; otherwise (None, None).
    """
    if target_num_layers == 0:
        return None, None

    if target_num_layers < 0:
        logger.warning(
            "Ignoring num_hidden_layers override: expected non-negative value, got %d.",
            target_num_layers,
        )
        return None, None

    if hasattr(hf_config, "num_hidden_layers"):
        original_num_layers = hf_config.num_hidden_layers
        decoder_cfg = hf_config
    else:
        text_cfg = getattr(hf_config, "text_config", None)
        if text_cfg is None or not hasattr(text_cfg, "num_hidden_layers"):
            raise AttributeError(
                f"{type(hf_config).__name__} has no decoder num_hidden_layers "
                "(expected on config or text_config)"
            )
        original_num_layers = text_cfg.num_hidden_layers
        decoder_cfg = text_cfg

    if target_num_layers < original_num_layers:
        decoder_cfg.num_hidden_layers = target_num_layers
        logger.info(
            "Overriding num_hidden_layers from %d to %d for debugging and testing purposes.",
            original_num_layers,
            target_num_layers,
        )
        return original_num_layers, target_num_layers

    return None, None
