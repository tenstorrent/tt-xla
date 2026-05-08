# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
import torch_xla
import torch_xla.runtime as xr
from infra import ComparisonConfig, Framework, Workload
from infra.testers.single_chip.op.op_tester import OpTester


class _SliceModel(torch.nn.Module):
    def forward(self, x):
        return x[:, :, -2:, :, :]


@pytest.mark.nightly
@pytest.mark.single_device
def test_slice():
    xr.set_device_type("TT")

    model = _SliceModel().eval()
    x = torch.randn(1, 16, 1, 60, 104, dtype=torch.bfloat16).to(torch_xla.device())

    tester = OpTester(comparison_config=ComparisonConfig(), framework=Framework.TORCH)
    workload = Workload(framework=Framework.TORCH, model=model, args=[x])
    tester.test(workload)
