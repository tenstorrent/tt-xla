# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
from infra import ComparisonConfig, Framework, Workload
from infra.testers.single_chip.op.op_tester import OpTester


class Sum(torch.nn.Module):
    def __init__(self, dim=None, keepdim=False):
        super().__init__()
        self.dim = dim
        self.keepdim = keepdim

    def forward(self, x):
        if self.dim is None:
            return x.sum()
        return x.sum(dim=self.dim, keepdim=self.keepdim)


@pytest.mark.parametrize(
    ["dim", "keepdim", "input_shape", "dtype"],
    [
        (None, False, (2,), torch.int64),
        (None, False, (4, 8), torch.int64),
        (None, False, (4, 8), torch.float32),
        (0, False, (4, 8), torch.bfloat16),
        (1, False, (4, 8), torch.float32),
        (-1, False, (4, 8), torch.bfloat16),
        (-2, False, (4, 8), torch.float32),
        (0, True, (4, 8), torch.bfloat16),
        ([0, 2], False, (4, 8, 16), torch.float32),
    ],
    ids=[
        "no_dim_int64",
        "no_dim_int64_2d",
        "no_dim_float32",
        "dim=0_bfloat16",
        "dim=1_float32",
        "dim=-1_bfloat16",
        "dim=-2_float32",
        "dim=0_keepdim=True_bfloat16",
        "dim=[0,2]_float32",
    ],
)
def test_sum(dim, keepdim, input_shape, dtype):

    model = Sum(dim=dim, keepdim=keepdim)
    if dtype == torch.int64:
        inputs = torch.randint(0, 2000, input_shape, dtype=torch.int64)
    else:
        inputs = torch.randn(input_shape, dtype=dtype)

    tester = OpTester(comparison_config=ComparisonConfig(), framework=Framework.TORCH)

    workload = Workload(
        framework=Framework.TORCH,
        model=model,
        args=[inputs],
    )

    tester.test(workload)
