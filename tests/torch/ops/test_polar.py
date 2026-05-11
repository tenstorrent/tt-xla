# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pytest
import torch
import torch_xla.runtime as xr
from infra import ComparisonConfig, Framework, Workload, run_op_test
from infra.testers.single_chip.op.op_tester import OpTester
from torch_xla.distributed.spmd import Mesh


@pytest.mark.xfail(
    reason="error: failed to legalize unresolved materialization from ('tensor<0x2xf64>') to ('tensor<0xcomplex<f64>>') that remained live after conversion — https://github.com/tenstorrent/tt-mlir/issues/8291"
)
def test_polar_sharded():
    class Polar(torch.nn.Module):
        def __init__(self):
            super().__init__()
            # make `angle` a parameter so we have something to attach a shard to
            self.angle = torch.nn.Parameter(
                torch.randn(1024, 22, dtype=torch.float64), requires_grad=False
            )

        def forward(self, abs_):
            return torch.polar(abs_, self.angle)

    model = Polar()

    num_devices = xr.global_runtime_device_count()
    mesh_shape = (1, num_devices)
    mesh = Mesh(np.array(range(num_devices)), mesh_shape, ("batch", "model"))

    def shard_spec_fn(model, args, kwargs):
        # shard the angle parameter on dim 0 across the "model" axis
        return {model.angle: ("model", None)}

    abs_ = torch.ones(1024, 22, dtype=torch.float64)

    run_op_test(
        model,
        [abs_],
        framework=Framework.TORCH,
        mesh=mesh,
        shard_spec_fn=shard_spec_fn,
    )
