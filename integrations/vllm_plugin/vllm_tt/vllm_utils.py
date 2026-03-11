# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Portions (c) 2026 Tenstorrent AI ULC

"""
Shared utilities for VLLM TT plugin integration.
"""

import contextlib

import torch


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

    class MockStream:
        def __init__(self, device_index=0):
            self.device = torch.device("cpu", device_index)

    def cpu_device_index():
        return 0

    def cpu_stream(device_index=0):
        return MockStream(device_index)

    # Store original functions
    original_get_device_index = getattr(torch._C, "_accelerator_getDeviceIndex", None)
    original_get_stream = getattr(torch._C, "_accelerator_getStream", None)

    try:
        # Patch functions to return CPU equivalents
        torch._C._accelerator_getDeviceIndex = cpu_device_index
        torch._C._accelerator_getStream = cpu_stream
        yield
    finally:
        # Restore original functions
        if original_get_device_index is not None:
            torch._C._accelerator_getDeviceIndex = original_get_device_index
        if original_get_stream is not None:
            torch._C._accelerator_getStream = original_get_stream
