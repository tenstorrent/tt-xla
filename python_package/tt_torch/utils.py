# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Portions (c) 2026 Tenstorrent AI ULC

"""
Shared utilities for TT-Torch backend integration.
"""

import contextlib
import re

import torch


class MockStream:
    """Mock stream class for JAX device compatibility."""

    def __init__(self, device_index=0):
        self.device = torch.device("cpu", device_index)


def cpu_device_index():
    """Return CPU device index for JAX compatibility."""
    return 0


def cpu_stream(device_index=0):
    """Return mock CPU stream for JAX compatibility."""
    return MockStream(device_index)


def apply_jax_compatibility_patches():
    """Apply JAX compatibility patches globally."""
    torch._C._accelerator_getDeviceIndex = cpu_device_index
    torch._C._accelerator_getStream = cpu_stream


def torch_version_at_least(major: int, minor: int) -> bool:
    """Return whether the current torch version is at least major.minor."""
    match = re.match(r"^(\d+)\.(\d+)", torch.__version__)
    if match is None:
        return False
    current_version = (int(match.group(1)), int(match.group(2)))
    return current_version >= (major, minor)


def is_torch_2_10_or_newer() -> bool:
    """Return whether the current torch runtime is 2.10 or newer."""
    return torch_version_at_least(2, 10)


@contextlib.contextmanager
def torch_dynamo_jax_compatibility():
    """
    Context manager to temporarily patch torch accelerator functions to be compatible with JAX devices.

    This fixes TorchDynamo compilation errors when using JAX/TT devices by temporarily patching:
    - torch._C._accelerator_getDeviceIndex() to return CPU device index
    - torch._C._accelerator_getStream() to return CPU stream

    TorchDynamo calls these functions during compilation but they fail with JAX devices,
    so we temporarily redirect them to CPU equivalents during compilation.
    """
    # Store original functions
    original_get_device_index = getattr(torch._C, "_accelerator_getDeviceIndex", None)
    original_get_stream = getattr(torch._C, "_accelerator_getStream", None)

    try:
        # Apply patches
        apply_jax_compatibility_patches()
        yield
    finally:
        # Restore original functions
        if original_get_device_index is not None:
            torch._C._accelerator_getDeviceIndex = original_get_device_index
        if original_get_stream is not None:
            torch._C._accelerator_getStream = original_get_stream
