# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
from typing import Any

from .logger import tt_init_logger

logger = tt_init_logger(__name__)


def determine_mesh_shape(
    num_devices: int, mesh_shape: tuple[int, int] | list[int] | None = None
) -> tuple[int, int]:
    """Resolve the (batch, model) SPMD mesh shape against the device count.

    When `mesh_shape` is None, defaults to a 1D mesh `(1, num_devices)`. When
    provided, it must have two positive dimensions whose product equals
    `num_devices`.
    """
    if mesh_shape is None:
        resolved = (1, num_devices)
        logger.info(f"Using default 1D mesh shape for {num_devices} devices: {resolved}")
        return resolved

    dims = list(mesh_shape)
    if len(dims) != 2:
        raise ValueError(
            f"mesh_shape must have exactly 2 dimensions (batch, model); got {mesh_shape}"
        )
    if any(d <= 0 for d in dims):
        raise ValueError(f"mesh_shape dimensions must be positive; got {mesh_shape}")
    if dims[0] * dims[1] != num_devices:
        raise ValueError(
            f"mesh_shape {tuple(dims)} has product {dims[0] * dims[1]}, "
            f"which does not match the device count {num_devices}"
        )
    resolved = (dims[0], dims[1])
    logger.info(f"Using mesh shape for {num_devices} devices: {resolved}")
    return resolved


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
