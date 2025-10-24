# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import os

import torch
import torch.nn as nn
import torch_xla
import torch_xla.core.xla_model as xm
import torch_xla.runtime as xr

import timm


def get_model():
    model = timm.create_model('ese_vovnet19b_dw.ra_in1k', pretrained=True)
    model = model.eval()


    return model


def dump_tensors():
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

    torch_xla.set_custom_compile_options(
        {
            "export_path": "vovnet",
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

    # Generate inputs.
    input = torch.rand((1, 3, 224, 224))

    # Connect the device.
    device = xm.xla_device()

    # Move inputs and model to device.
    input = input.to(device)
    model = model.to(device)

    torch_xla.set_custom_compile_options(
        {
            "export_path": "vovnet",
            "backend": "codegen_py",
        }
    )
    output = model(input)
    # output = model(input).to("cpu")


def run_on_cpu():
    model = get_model()

    # Generate inputs.
    input = torch.rand((1, 3, 224, 224))

    # Connect the device.
    device = xm.xla_device()

    # Move inputs and model to device.
    input = input.to(device)
    model = model.to(device)

    output = model(input)
    print(output)


# --------------------------------
# main
# --------------------------------
if __name__ == "__main__":
    # print("Dumping code...")
    # dump_code()
    print("Dumping tensors...")
    dump_tensors()
    # print("Running on cpu...")
    # run_on_cpu()
