# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import os

import torch
import torch.nn as nn
import torch_xla.core.xla_model as xm
import torch_xla.runtime as xr


# --------------------------------
# Test run
# --------------------------------
def alexnet():
    # Instantiate model.
    model: nn.Module = torch.hub.load(
        "pytorch/vision:v0.10.0", "alexnet", pretrained=True
    )

    # Put it in inference mode and compile it.
    model = model.eval()
    model.compile(backend="tt")

    # Generate inputs.
    input = torch.rand((1, 3, 224, 224))

    # Connect the device.
    device = xm.xla_device()

    # Move inputs and model to device.
    input = input.to(device)
    model = model.to(device)

    # Run model (with no gradient calculation since we only need inference).
    with torch.no_grad():
        output = model(input)

    print(output)


# --------------------------------
# main
# --------------------------------
if __name__ == "__main__":
    # By default torch_xla uses the CPU device so we have to set it to TT device.
    xr.set_device_type("TT")

    alexnet()
