# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import torch
import torch_xla.core.xla_model as xm
import torch_xla.runtime as xr

from third_party.tt_forge_models.mnist.image_classification.pytorch.loader import (
    MNISTCNNDropoutModel,
)


# --------------------------------
# Test run
# --------------------------------
def run_mnist():
    """Run MNIST model on TT device."""
    # Instantiate model.
    torch.manual_seed(42)
    model: torch.nn.Module = MNISTCNNDropoutModel().to(dtype=torch.bfloat16)

    # Put it in inference mode and compile it.
    model = model.eval()
    model.compile(backend="tt")

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

    return output


def run_mnist_cpu():
    """Run MNIST model on CPU for comparison."""
    # Instantiate model with same seed.
    torch.manual_seed(42)
    model: torch.nn.Module = MNISTCNNDropoutModel().to(dtype=torch.bfloat16)

    # Put it in inference mode.
    model = model.eval()

    # Generate inputs.
    input = torch.ones((4, 1, 28, 28), dtype=torch.bfloat16)

    # Run model on CPU.
    with torch.no_grad():
        output = model(input)

    return output


def test_mnist():
    """Test MNIST TT output against CPU reference."""
    xr.set_device_type("TT")

    tt_output = run_mnist()
    cpu_output = run_mnist_cpu()

    tt_output_cpu = tt_output.cpu()

    tt_flat = tt_output_cpu.flatten().float()
    cpu_flat = cpu_output.flatten().float()
    pcc = torch.corrcoef(torch.stack([tt_flat, cpu_flat]))[0, 1].item()

    print(f"PCC: {pcc}")
    print(f"Max diff: {(tt_output_cpu - cpu_output).abs().max()}")

    assert pcc > 0.99, f"PCC too low: {pcc}, expected > 0.99"


# --------------------------------
# main
# --------------------------------
if __name__ == "__main__":
    # By default torch_xla uses the CPU device so we have to set it to TT device.
    xr.set_device_type("TT")

    output = run_mnist()
    print(output)
