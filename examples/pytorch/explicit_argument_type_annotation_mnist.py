# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import torch
import torch_xla.core.xla_model as xm
import torch_xla.runtime as xr
import tt_torch_backend as tt_torch

from tests.torch.single_chip.models.mnist.cnn.dropout.model_implementation import (
    MNISTCNNDropoutModel,
)


# --------------------------------
# Test run
# --------------------------------
def mnist_with_consteval():
    # Instantiate model.
    model: torch.nn.Module = MNISTCNNDropoutModel().to(dtype=torch.bfloat16)

    # Put it in inference mode.
    model = model.eval()

    # Generate inputs.
    input = torch.ones((4, 1, 28, 28), dtype=torch.bfloat16)

    # Connect the device.
    device = xm.xla_device()

    # Move inputs and model to device.
    input = input.to(device)
    model = model.to(device)

    # Mark model inputs before compiling. This will allow the compiler to
    # distinguish between input and parameter arguments, enabling consteval and other optimizations.
    tt_torch.mark_module_user_inputs(model)

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

    mnist_with_consteval()
