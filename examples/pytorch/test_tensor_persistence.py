# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
import os
import torch
import torch_xla
import torch_xla.runtime as xr
import torch_xla.core.xla_model as xm
import torch_xla.distributed.spmd as xs
from torch_xla.distributed.spmd import Mesh


def test_add():
    class AddModel(torch.nn.Module):
        def __init__(self):
            super().__init__()

        def forward(self, x):
            return x + 1

    xr.set_device_type("TT")
    device = torch_xla.device()

    x: torch.Tensor = torch.ones((3, 3)).to(device)
    model = AddModel()
    model = model.to(device)
    model.compile(backend="tt")
    y = model(x)

    print(y)


def test_add_sharded():
    class AddModel(torch.nn.Module):
        def __init__(self):
            super().__init__()

        def forward(self, x):
            x += 1
            return x

    xr.set_device_type("TT")
    xr.use_spmd()

    device = xm.xla_device()
    mesh: Mesh = Mesh(
        list(range(xr.global_runtime_device_count())),
        (1, xr.global_runtime_device_count()),
        ("batch", "model"),
    )

    x: torch.Tensor = torch.arange(16).reshape((4, 4)).to(device)
    xs.mark_sharding(x, mesh, ("model", None))

    model = AddModel()
    model = model.to(device)
    model.compile(backend="tt")
    x = model(x)
    print(x)
