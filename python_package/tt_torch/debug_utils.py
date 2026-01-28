# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
Debugging utilities for detecting FakeTensors and tracing their origin.
"""
import logging
import traceback
from typing import Any, Optional

import torch

logger = logging.getLogger(__name__)


def is_fake_tensor(tensor: torch.Tensor) -> bool:
    """Check if a tensor is a FakeTensor."""
    return hasattr(tensor, "_is_fake") and tensor._is_fake


def get_tensor_info(tensor: torch.Tensor, name: str = "tensor") -> str:
    """Get detailed information about a tensor for debugging."""
    info = [
        f"{name}:",
        f"  type: {type(tensor)}",
        f"  shape: {tensor.shape}",
        f"  dtype: {tensor.dtype}",
        f"  device: {tensor.device}",
        f"  is_fake: {is_fake_tensor(tensor)}",
    ]

    if is_fake_tensor(tensor):
        info.append(f"  _is_fake: {tensor._is_fake}")
        if hasattr(tensor, "_fake_mode"):
            info.append(f"  _fake_mode: {tensor._fake_mode}")
        if hasattr(tensor, "_fake_mode_id"):
            info.append(f"  _fake_mode_id: {tensor._fake_mode_id}")

    return "\n".join(info)


def check_and_log_fake_tensor(
    tensor: torch.Tensor,
    context: str,
    name: Optional[str] = None,
    raise_error: bool = False,
) -> bool:
    """
    Check if a tensor is a FakeTensor and log detailed information if it is.

    Args:
        tensor: The tensor to check
        context: Description of where this check is happening
        name: Optional name for the tensor
        raise_error: If True, raise an error when FakeTensor is detected

    Returns:
        True if tensor is a FakeTensor, False otherwise
    """
    if is_fake_tensor(tensor):
        tensor_name = name or "tensor"
        logger.error(
            f"FakeTensor detected in {context}!\n"
            f"{get_tensor_info(tensor, tensor_name)}\n"
            f"Stack trace:\n{''.join(traceback.format_stack()[:-1])}"
        )
        if raise_error:
            raise RuntimeError(
                f"FakeTensor detected in {context}: {tensor_name} "
                f"(shape={tensor.shape}, device={tensor.device})"
            )
        return True
    return False


def check_tensors_in_state_dict(
    state_dict: dict[str, torch.Tensor],
    context: str,
    raise_error: bool = False,
) -> list[str]:
    """
    Check all tensors in a state dict for FakeTensors.

    Returns:
        List of keys that contain FakeTensors
    """
    fake_keys = []
    for key, tensor in state_dict.items():
        if isinstance(tensor, torch.Tensor) and is_fake_tensor(tensor):
            fake_keys.append(key)
            logger.error(
                f"FakeTensor in state_dict[{key}] in {context}!\n"
                f"{get_tensor_info(tensor, key)}"
            )
            if raise_error:
                raise RuntimeError(f"FakeTensor in state_dict[{key}] in {context}")
    return fake_keys


def check_model_parameters(
    model: torch.nn.Module,
    context: str,
    raise_error: bool = False,
) -> list[str]:
    """
    Check all parameters and buffers in a model for FakeTensors.

    Returns:
        List of parameter/buffer names that are FakeTensors
    """
    fake_names = []
    for name, param in model.named_parameters():
        if is_fake_tensor(param):
            fake_names.append(f"parameter:{name}")
            logger.error(
                f"FakeTensor parameter '{name}' in {context}!\n"
                f"{get_tensor_info(param, name)}"
            )
            if raise_error:
                raise RuntimeError(f"FakeTensor parameter '{name}' in {context}")

    for name, buffer in model.named_buffers():
        if is_fake_tensor(buffer):
            fake_names.append(f"buffer:{name}")
            logger.error(
                f"FakeTensor buffer '{name}' in {context}!\n"
                f"{get_tensor_info(buffer, name)}"
            )
            if raise_error:
                raise RuntimeError(f"FakeTensor buffer '{name}' in {context}")

    return fake_names


def check_fake_tensor_mode_active(context: str) -> bool:
    """
    Check if FakeTensorMode is currently active.

    Returns:
        True if FakeTensorMode is active, False otherwise
    """
    try:
        from torch.utils._python_dispatch import _get_current_dispatch_mode

        current_mode = _get_current_dispatch_mode()
        if current_mode is not None:
            mode_type = type(current_mode).__name__
            if "FakeTensor" in mode_type:
                logger.warning(
                    f"FakeTensorMode is ACTIVE in {context}!\n"
                    f"Mode type: {mode_type}\n"
                    f"Mode object: {current_mode}\n"
                    f"Stack trace:\n{''.join(traceback.format_stack()[:-1])}"
                )
                return True
    except (ImportError, AttributeError):
        pass
    return False
