# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
from typing import Any

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
