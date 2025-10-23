# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pytest
import torch
import torch_xla.runtime as xr
from infra import Framework, run_graph_test, run_graph_test_with_random_inputs
from torch_xla.distributed.spmd import Mesh
from utils import Category


@pytest.mark.push
@pytest.mark.nightly
@pytest.mark.record_test_properties(category=Category.GRAPH_TEST)
@pytest.mark.parametrize("bias", [True, False])
def test_simple_matmul(bias):
    class MatMul(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = torch.nn.Linear(32, 64, bias=bias, dtype=torch.bfloat16)

        def forward(self, x):
            return self.linear(x)

    model = MatMul()
    run_graph_test_with_random_inputs(
        model, [(32, 32)], dtype=torch.bfloat16, framework=Framework.TORCH
    )


@pytest.mark.record_test_properties(category=Category.GRAPH_TEST)
def test_matmul_correct_sharding():
    class MatMul(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.linear_a = torch.nn.Linear(2048, 2048, bias=True, dtype=torch.bfloat16)
            self.linear_b = torch.nn.Linear(2048, 2048, bias=True, dtype=torch.bfloat16)

        def forward(self, x):
            x = self.linear_a(x)
            x = self.linear_b(x)
            x = torch.relu(x)
            return x

    num_devices = xr.global_runtime_device_count()
    mesh_shape = (1, num_devices)
    device_ids = np.array(range(num_devices))
    mesh = Mesh(device_ids, mesh_shape, ("batch", "model"))

    def get_shard_spec(matmul):
        shard_specs = {}
        shard_specs[matmul.linear_a.weight] = ("model", None)
        shard_specs[matmul.linear_a.bias] = ("model",)
        shard_specs[matmul.linear_b.weight] = (None, "model")
        return shard_specs

    x = torch.randn(2048, 2048, dtype=torch.bfloat16)
    model = MatMul()
    run_graph_test(
        model, [x], framework=Framework.TORCH, mesh=mesh, shard_spec_fn=get_shard_spec
    )


@pytest.mark.record_test_properties(category=Category.GRAPH_TEST)
def test_matmul_incorrect_sharding():
    class MatMul(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.linear_a = torch.nn.Linear(2048, 2048, bias=True, dtype=torch.bfloat16)
            self.linear_b = torch.nn.Linear(2048, 2048, bias=True, dtype=torch.bfloat16)

        def forward(self, x):
            x = self.linear_a(x)
            x = torch.relu(x)
            x = self.linear_b(x)
            x = torch.relu(x)
            return x

    num_devices = xr.global_runtime_device_count()
    mesh_shape = (1, num_devices)
    device_ids = np.array(range(num_devices))
    mesh = Mesh(device_ids, mesh_shape, ("batch", "model"))

    def get_shard_spec(matmul):
        shard_specs = {}
        shard_specs[matmul.linear_a.weight] = ("model", None)
        shard_specs[matmul.linear_a.bias] = ("model",)
        shard_specs[matmul.linear_b.weight] = (None, "model")
        return shard_specs

    x = torch.randn(2048, 2048, dtype=torch.bfloat16)
    model = MatMul()
    run_graph_test(
        model, [x], framework=Framework.TORCH, mesh=mesh, shard_spec_fn=get_shard_spec
    )
