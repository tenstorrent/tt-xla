# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pytest
import torch
import torch_xla.runtime as xr
from infra import Framework, run_graph_test
from torch_xla.distributed.spmd import Mesh
from utils import Category

from tests.utils import parametrize_arch


@pytest.mark.extended
@pytest.mark.nightly
@pytest.mark.record_test_properties(category=Category.GRAPH_TEST)
@pytest.mark.filecheck(["distributed_rms_norm.ttir.mlir"])
@parametrize_arch(["dual_chip", "llmbox", "galaxy"])
def test_distributed_rmsnorm(arch, request):
    """Test distributed RMS norm fusion with normalized dimension sharded.

    When the normalized (hidden) dimension is sharded across the model axis,
    Shardy inserts all_gather before and all_slice after the RMS norm custom
    call. The FuseDistributedCustomCallsPass fuses these into a single
    distributed_rms_norm op (tenstorrent/tt-mlir#7878).
    """

    class RMSNorm(torch.nn.Module):
        def __init__(self, hidden_size, eps=1e-6):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.ones(hidden_size))
            self.hidden_size = hidden_size
            self.eps = eps

        def forward(self, x):
            return torch.nn.functional.rms_norm(
                x, (self.hidden_size,), self.weight, self.eps
            )

    hidden_size = 256
    model = RMSNorm(hidden_size).to(torch.bfloat16)
    x = torch.randn(8, hidden_size, dtype=torch.bfloat16)

    num_devices = xr.global_runtime_device_count()
    mesh = Mesh(np.array(range(num_devices)), (1, num_devices), ("batch", "model"))

    def shard_spec_fn(model, args, kwargs):
        return {args[0]: (None, "model")}

    run_graph_test(
        model,
        [x],
        framework=Framework.TORCH,
        mesh=mesh,
        shard_spec_fn=shard_spec_fn,
        request=request,
    )


@pytest.mark.extended
@pytest.mark.nightly
@pytest.mark.record_test_properties(category=Category.GRAPH_TEST)
@pytest.mark.filecheck(["distributed_rms_norm.ttir.mlir"])
@parametrize_arch(["dual_chip", "llmbox", "galaxy"])
def test_distributed_rmsnorm_3d(arch, request):
    """Test distributed RMS norm fusion with a 3D input (batch, seq, hidden).

    Mirrors the typical transformer hidden state shape where RMS norm is
    applied over the last dimension. The hidden dimension is sharded across
    the model axis to trigger the distributed fusion.
    """

    class RMSNorm(torch.nn.Module):
        def __init__(self, hidden_size, eps=1e-6):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.ones(hidden_size))
            self.hidden_size = hidden_size
            self.eps = eps

        def forward(self, x):
            return torch.nn.functional.rms_norm(
                x, (self.hidden_size,), self.weight, self.eps
            )

    hidden_size = 256
    model = RMSNorm(hidden_size).to(torch.bfloat16)
    x = torch.randn(2, 32, hidden_size, dtype=torch.bfloat16)

    num_devices = xr.global_runtime_device_count()
    mesh = Mesh(np.array(range(num_devices)), (1, num_devices), ("batch", "model"))

    def shard_spec_fn(model, args, kwargs):
        return {args[0]: (None, None, "model")}

    run_graph_test(
        model,
        [x],
        framework=Framework.TORCH,
        mesh=mesh,
        shard_spec_fn=shard_spec_fn,
        request=request,
    )
