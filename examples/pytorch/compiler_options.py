# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
Example demonstrating the compiler configuration options for TT-XLA with PyTorch.
"""

import torch
import torch_xla
import torch_xla.core.xla_model as xm
import torch_xla.runtime as xr

from tests.torch.models.mnist.cnn.dropout.model_implementation import (
    MNISTCNNDropoutModel,
)


# --------------------------------
# Test run
# --------------------------------
def mnist_with_compiler_options():
    """Test the compiler configuration options with MNIST model."""

    print("Testing compiler configuration options with MNIST CNN model...\n")

    device = xm.xla_device()

    options = {
        "optimization_level": 2,
    }

    # Set compile options
    torch_xla.set_custom_compile_options(options)

    # Create model
    model = MNISTCNNDropoutModel().to(dtype=torch.bfloat16)
    model = model.eval()

    # Create test input
    input_tensor = torch.ones((4, 1, 28, 28), dtype=torch.bfloat16)

    # Move to device
    model = model.to(device)
    input_device = input_tensor.to(device)

    # Compile model
    model.compile(backend="tt")

    # Run inference
    print("Running inference with compiler options...")
    with torch.no_grad():
        output = model(input_device)

    print(f"Success! Output shape: {output.shape}")


if __name__ == "__main__":
    # Set device to TT
    xr.set_device_type("TT")

    mnist_with_compiler_options()
