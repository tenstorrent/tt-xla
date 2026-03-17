# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Minimal torch.empty empty-tensor host-transfer repro."""

import pytest
import torch
import torch_xla.core.xla_model as xm
import torch_xla.runtime as xr
from utils import Category


@pytest.mark.nightly
@pytest.mark.single_device
@pytest.mark.record_test_properties(
    category=Category.OP_TEST,
    torch_op_name="torch.empty",
)
def test_empty():
    class Empty(torch.nn.Module):
        def forward(self, x):
            return torch.empty((0,), device=x.device, dtype=torch.int64)

    xr.set_device_type("TT")

    model = torch.compile(Empty(), backend="tt")
    device = xm.xla_device()
    model = model.to(device)
    input_tensor = torch.tensor(0, dtype=torch.int64).to(device)

    with torch.no_grad():
        output = model(input_tensor)

    host_output = output.to("cpu")
