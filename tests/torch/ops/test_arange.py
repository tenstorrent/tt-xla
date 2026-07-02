# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""torch.arange no-input-graph sharding repro: it const-folds to a no-argument graph
that collapses the mesh [1,4]->[1,1] while a sharded buffer is live -> device mismatch.
"""

import numpy as np
import pytest
import torch
import torch_xla
import torch_xla.runtime as xr
from infra import Framework, run_op_test
from torch_xla.distributed.spmd import Mesh
from utils import Category

DIM = 64


class Arange(torch.nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.proj = torch.nn.Linear(
            dim, dim, bias=False, dtype=torch.bfloat16
        )  # sharded buffer

    def forward(self, x):
        return torch.arange(0, DIM, dtype=torch.float32, device=x.device)


@pytest.mark.nightly
@pytest.mark.llmbox
@pytest.mark.record_test_properties(
    category=Category.OP_TEST, torch_op_name="torch.arange"
)
def test_arange_no_input_graph_sharding():
    model = Arange(DIM)
    num_devices = xr.global_runtime_device_count()
    mesh = Mesh(np.array(range(num_devices)), (1, num_devices), ("batch", "model"))

    run_op_test(
        model,
        [torch.randn(1, DIM, dtype=torch.bfloat16)],
        framework=Framework.TORCH,
        mesh=mesh,
        shard_spec_fn=lambda m: {m.proj.weight: ("model", "batch")},
    )
