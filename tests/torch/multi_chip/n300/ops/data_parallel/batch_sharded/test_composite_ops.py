# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
import os

import numpy as np
import pytest
import torch
import torch_xla
import torch_xla.distributed.spmd as xs
import torch_xla.runtime as xr
from infra.comparators.torch_comparator import TorchComparator

from tests.infra.comparators.comparison_config import ComparisonConfig


@pytest.mark.parametrize("use_weight", [True, False])
def test_rmsnorm(use_weight):

    class RMSNormModel(torch.nn.Module):
        def __init__(self, normalized_shape):
            super().__init__()
            self.normalized_shape = normalized_shape

        def forward(self, x, weight):
            return torch.nn.functional.rms_norm(x, self.normalized_shape, weight)

    # Set SPMD mode and get number of devices.
    os.environ["CONVERT_SHLO_TO_SHARDY"] = "1"
    xr.use_spmd()

    num_devices = xr.global_runtime_device_count()

    # Create a mesh.
    mesh_shape = (1, num_devices)
    device_ids = np.array(range(num_devices))
    mesh = xs.Mesh(device_ids, mesh_shape, ("model", "batch"))

    options = {"tt_enable_composite_ops": True}

    normalized_shape = (32,)
    input_shape = (4, 32)
    input_tensor = torch.randn(input_shape)

    weight = torch.randn(normalized_shape) if use_weight else None

    model = RMSNormModel(normalized_shape)
    golden = model(input_tensor, weight if use_weight else None)

    device = torch_xla.device()
    model = torch.compile(model.to(device), backend="tt", options=options)

    # Mark sharding for inputs along batch dimension.
    input_tensor = input_tensor.to(device)
    xs.mark_sharding(input_tensor, mesh, ("batch", None))

    if use_weight:
        weight = weight.to(device)
        xs.mark_sharding(weight, mesh, (None,))

    output = model(input_tensor, weight if use_weight else None)

    comparator = TorchComparator(ComparisonConfig())
    comparator.compare(output, golden)
