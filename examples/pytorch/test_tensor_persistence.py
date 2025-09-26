# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
import pytest
import torch
import torch_xla.core.xla_model as xm
import torch_xla.runtime as xr
import torch_xla


def test_inplace_add_multiloop():
    class AddModule(torch.nn.Module):
        def __init__(self):
            super().__init__()

        def forward(self, x):
            x = x + 1
            return x

    # By default torch_xla uses the CPU device so we have to set it to TT device.
    xr.set_device_type("TT")

    # set up inputs and model
    x = torch.zeros((3, 3), dtype=torch.bfloat16)

    model = AddModule()
    model.compile(backend="tt")

    output = None
    n_loops = 3

    # Connect the device.
    device = xm.xla_device()

    # Move inputs and model to device.
    x = x.to(device)
    model = model.to(device)

    # compile the model

    with torch.no_grad():
        for _ in range(n_loops):
            x = model(x)
            print(x)

    # result = x.to("cpu")
    # assert result.equal(torch.ones(3,3)*n_loops)


def test_pure_multiloop():
    class AddModule(torch.nn.Module):
        def __init__(self):
            super().__init__()

        def forward(self, x):
            y = x + 1
            return y

    # By default torch_xla uses the CPU device so we have to set it to TT device.
    xr.set_device_type("TT")

    # set up inputs and model
    x = torch.zeros((3, 3), dtype=torch.bfloat16)

    model = AddModule()
    model.compile(backend="tt")

    output = None
    n_loops = 3

    # Connect the device.
    device = xm.xla_device()

    # Move inputs and model to device.
    x = x.to(device)
    model = model.to(device)

    # compile the model

    with torch.no_grad():
        for _ in range(n_loops):
            y = model(x)
            print(y)

    # result = x.to("cpu")
    # assert result.equal(torch.ones(3,3)*n_loops)
