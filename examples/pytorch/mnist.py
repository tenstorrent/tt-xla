# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import os

import torch
import torch_xla.core.xla_model as xm
from torch_xla.experimental import plugins

from tests.torch.single_chip.models.mnist.cnn.dropout.model_implementation import (
    MNISTCNNDropoutModel,
)

# --------------------------------
# Plugin registration
# --------------------------------
os.environ["PJRT_DEVICE"] = "TT"
os.environ["XLA_STABLEHLO_COMPILE"] = "1"


class TTPjrtPlugin(plugins.DevicePlugin):
    def library_path(self):
        return os.path.join(os.getcwd(), "build/src/tt/pjrt_plugin_tt.so")


plugins.register_plugin("TT", TTPjrtPlugin())


# --------------------------------
# Test run
# --------------------------------
def mnist():
    # Instantiate model.
    model: torch.nn.Module = MNISTCNNDropoutModel().to(dtype=torch.bfloat16)

    # Put it in inference mode and compile it.
    model = model.eval()
    model.compile(backend="openxla")

    # Generate inputs.
    input = torch.ones((4, 1, 28, 28), dtype=torch.bfloat16)

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
    mnist()
