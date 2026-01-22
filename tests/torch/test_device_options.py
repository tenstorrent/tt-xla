# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
Simple test for device options functionality.

Run with TTXLA_LOGGER_LEVEL=DEBUG to see the device options being received
in tt-xla logs.
"""

import torch
import torch_xla
import torch_xla.core.xla_model as xm


def test_device_options():
    """Set device options and run a simple computation."""
    # Set device options for device 0
    torch_xla.set_custom_device_options(
        device_id=0,
        options={
            "l1SmallSize": "16384",
            "testOption": "testValue",
        },
    )

    # Simple computation to trigger execution
    device = xm.xla_device()
    x = torch.randn(4, 4, dtype=torch.bfloat16).to(device)
    y = torch.randn(4, 4, dtype=torch.bfloat16).to(device)
    z = (x + y).to("cpu")

    print(f"Result shape: {z.shape}")
    print("Device options test completed - check tt-xla logs for options")


if __name__ == "__main__":
    test_device_options()
