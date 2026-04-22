# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import torch
from infra import ComparisonConfig, Framework, Workload
from infra.testers.single_chip.op.op_tester import OpTester
from torch import nn


def test_interpolate():

    class Interpolate(torch.nn.Module):
        def forward(self, spatial_pos, h, w):
            return nn.functional.interpolate(
                spatial_pos,
                size=(h, w),
                mode="bilinear",
                align_corners=True,
            )

    model = Interpolate().eval()

    spatial_pos = torch.randn(1, 768, 12, 12, dtype=torch.bfloat16)
    h = torch.tensor(12, dtype=torch.int64)
    w = torch.tensor(16, dtype=torch.int64)

    tester = OpTester(comparison_config=ComparisonConfig(), framework=Framework.TORCH)
    workload = Workload(
        framework=Framework.TORCH,
        model=model,
        args=[spatial_pos, h, w],
    )
    tester.test(workload)
