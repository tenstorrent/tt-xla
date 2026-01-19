# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

### Demonstrates VoVNet from timm (ese_vovnet19b_dw.ra_in1k)
### Supports two modes: codegen and run on TT device

import argparse

import timm
import torch
import torch.nn as nn
import torch_xla
import torch_xla.core.xla_model as xm
import torch_xla.runtime as xr


def run_codegen(model, activation_tensor):
    """Generate Python code from the model."""
    from tt_torch import codegen_py

    extra_options = {
        "optimization_level": 2,
    }

    codegen_py(
        model,
        activation_tensor,
        export_path="vovnet_codegen",
        compiler_options=extra_options,
    )


def run_on_device(model, activation_tensor):
    """Run the model on TT device."""
    options = {
        "optimization_level": 2,
    }
    torch_xla.set_custom_compile_options(options)

    model.compile(backend="tt")

    device = xm.xla_device()

    input = activation_tensor.to(device)
    model = model.to(device)

    with torch.no_grad():
        output = model(input)

    print(output)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="VoVNet example")
    parser.add_argument(
        "--mode",
        choices=["codegen", "run"],
        default="run",
        help="Mode: 'codegen' to generate code, 'run' to execute on TT device",
    )
    args = parser.parse_args()

    # Set up XLA runtime for TT backend
    xr.set_device_type("TT")

    # Load VoVNet from timm
    model = timm.create_model("ese_vovnet19b_dw.ra_in1k", pretrained=True)
    model.eval()

    activation_tensor = torch.randn(1, 3, 224, 224)

    if args.mode == "codegen":
        run_codegen(model, activation_tensor)
    else:
        run_on_device(model, activation_tensor)
