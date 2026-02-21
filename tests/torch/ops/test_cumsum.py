# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
from infra import Framework, run_op_test
from utils import Category


class CumsumOp(torch.nn.Module):
    def __init__(self, dim=0):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        return torch.cumsum(x, dim=self.dim)


@pytest.mark.push
@pytest.mark.nightly
@pytest.mark.single_device
@pytest.mark.record_test_properties(
    category=Category.OP_TEST,
    torch_op_name="torch.cumsum",
)
@pytest.mark.parametrize(
    "size",
    [
        # 1024, PASSES
        # 32768, PASSES
        1168640, # deepseek_ocr model input shape
    ],
    ids=lambda val: f"size_{val}",
)
def test_cumsum_1d_int64(size):
    input_tensor = torch.randint(0, 2, (size,), dtype=torch.int64)
    run_op_test(
        CumsumOp(dim=0),
        [input_tensor],
        framework=Framework.TORCH,
    )
