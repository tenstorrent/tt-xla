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

from third_party.tt_forge_models.mnist.image_classification.pytorch.loader import (
    MNISTCNNDropoutModel,
)


# --------------------------------
# Test run
# --------------------------------
def run_mnist_with_compiler_options():
    """Run MNIST model with compiler options on TT device."""
    device = xm.xla_device()

    options = {
        "optimization_level": 2,
    }

    # Set compile options
    torch_xla.set_custom_compile_options(options)

    # Create model
    torch.manual_seed(42)
    model = MNISTCNNDropoutModel().to(dtype=torch.bfloat16)
    model = model.eval()

    # Create test input
    torch.manual_seed(42)
    input_tensor = torch.ones((4, 1, 28, 28), dtype=torch.bfloat16)

    # Move to device
    model = model.to(device)
    input_device = input_tensor.to(device)

    # Compile model
    model.compile(backend="tt")

    # Run inference
    with torch.no_grad():
        output = model(input_device)

    return output


def run_mnist_cpu():
    """Run MNIST model on CPU for comparison."""
    # Create model with same seed
    torch.manual_seed(42)
    model = MNISTCNNDropoutModel().to(dtype=torch.bfloat16)
    model = model.eval()

    # Create test input with same seed
    torch.manual_seed(42)
    input_tensor = torch.ones((4, 1, 28, 28), dtype=torch.bfloat16)

    # Run inference on CPU
    with torch.no_grad():
        output = model(input_tensor)

    return output


def test_compiler_options():
    """Test MNIST with compiler options TT output against CPU reference."""
    xr.set_device_type("TT")

    tt_output = run_mnist_with_compiler_options()
    cpu_output = run_mnist_cpu()

    tt_output_cpu = tt_output.cpu()

    tt_flat = tt_output_cpu.flatten().float()
    cpu_flat = cpu_output.flatten().float()
    pcc = torch.corrcoef(torch.stack([tt_flat, cpu_flat]))[0, 1].item()

    print(f"PCC: {pcc}")
    print(f"Max diff: {(tt_output_cpu - cpu_output).abs().max()}")

    assert pcc > 0.99, f"PCC too low: {pcc}, expected > 0.99"


if __name__ == "__main__":
    # Set device to TT
    xr.set_device_type("TT")

    output = run_mnist_with_compiler_options()
    print(f"Success! Output shape: {output.shape}")
