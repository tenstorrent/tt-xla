# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import os

import torch
import torch.nn as nn
import torch_xla
import torch_xla.core.xla_model as xm
import torch_xla.runtime as xr

from transformers import ResNetForImageClassification


def get_model():
    # Download model.
    # model: nn.Module = torch.hub.load(
    #     "pytorch/vision:v0.10.0", "alexnet", pretrained=True
    # )
    # Load ResNet-50 from HuggingFace
    model = ResNetForImageClassification.from_pretrained("microsoft/resnet-50")

    # Put it in inference mode.
    model = model.eval()

    return model

def get_input():
    torch.manual_seed(4)
    input = torch.rand((1, 3, 224, 224))

    return input


def dump_tensors():
    xr.set_device_type("TT")

    model = get_model()
    model.compile(backend="tt")

    input = get_input()

    # Connect the device.
    device = xm.xla_device()

    # Move inputs and model to device.
    input = input.to(device)
    model = model.to(device)

    torch_xla.set_custom_compile_options(
        {
            "export_path": "resnet",
            "dump_inputs": True,
        }
    )
    output = model(input)
    # output = model(input).to("cpu")
    return


def dump_code():
    xr.set_device_type("TT")

    model = get_model()

    model.compile(backend="tt")

    input = get_input()

    # Connect the device.
    device = xm.xla_device()

    # Move inputs and model to device.
    input = input.to(device)
    model = model.to(device)

    torch_xla.set_custom_compile_options(
        {
            "export_path": "resnet",
            "backend": "codegen_py",
        }
    )
    output = model(input)
    # output = model(input).to("cpu")


def run_on_cpu():
    model = get_model()

    input = get_input()

    # Connect the device.
    device = xm.xla_device()

    # Move inputs and model to device.
    input = input.to(device)
    model = model.to(device)

    output = model(input)
    print(output)


def run_on_tt():
    xr.set_device_type("TT")

    model = get_model()

    model.compile(backend="tt")

    # Generate inputs.
    input = torch.rand((1, 3, 224, 224))

    # Connect the device.
    device = xm.xla_device()

    # Move inputs and model to device.
    input = input.to(device)
    model = model.to(device)

    output = model(input)
    print(output)
    return


# --------------------------------
# main
# --------------------------------
if __name__ == "__main__":
    print("Dumping code...")
    dump_code()

    # print("Dumping tensors...")
    # dump_tensors()

    # print("Running on cpu...")
    # run_on_cpu()

    # print("Running on tt...")
    # run_on_tt()
