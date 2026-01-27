# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import torch
import torch.nn as nn
import torch_xla.core.xla_model as xm
import torch_xla.runtime as xr
from torchvision.models import alexnet as alexnet_model


# --------------------------------
# Test run
# --------------------------------
def run_alexnet():
    """Run AlexNet on TT device."""
    # Instantiate model.
    model: nn.Module = alexnet_model(pretrained=True)

    # Put it in inference mode and compile it.
    model = model.eval()
    model.compile(backend="tt")

    # Generate inputs with fixed seed for reproducibility.
    torch.manual_seed(42)
    input = torch.rand((1, 3, 224, 224))

    # Connect the device.
    device = xm.xla_device()

    # Move inputs and model to device.
    input = input.to(device)
    model = model.to(device)

    # Run model (with no gradient calculation since we only need inference).
    with torch.no_grad():
        output = model(input)

    return output


def run_alexnet_cpu():
    """Run AlexNet on CPU for comparison."""
    # Instantiate model.
    model: nn.Module = alexnet_model(pretrained=True)
    model = model.eval()

    # Generate inputs with same fixed seed.
    torch.manual_seed(42)
    input = torch.rand((1, 3, 224, 224))

    # Run model on CPU.
    with torch.no_grad():
        output = model(input)

    return output


def test_alexnet():
    """Test AlexNet TT output against CPU reference."""
    xr.set_device_type("TT")

    tt_output = run_alexnet()
    cpu_output = run_alexnet_cpu()

    tt_output_cpu = tt_output.cpu()

    tt_flat = tt_output_cpu.flatten()
    cpu_flat = cpu_output.flatten()
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

    output = run_alexnet()
    print(output)
