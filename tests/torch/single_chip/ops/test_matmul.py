# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
from infra import Framework, run_op_test_with_random_inputs
from utils import Category

from tests.infra.testers.compiler_config import CompilerConfig


class Matmul(torch.nn.Module):
    def __init__(self, inner_dim, rhs_outer_dim, dtype=torch.float32):
        super().__init__()
        self.weight = torch.nn.Parameter(
            torch.randn(inner_dim, rhs_outer_dim, dtype=dtype)
        )

    def forward(self, x):
        return torch.matmul(x, self.weight)


@pytest.mark.push
@pytest.mark.nightly
@pytest.mark.record_test_properties(category=Category.OP_TEST)
@pytest.mark.parametrize("lhs_outer", [32, 64])
@pytest.mark.parametrize("rhs_outer", [32, 64])
@pytest.mark.parametrize("inner", [32, 64])
@pytest.mark.parametrize("experimental_bfp8_weights", [False, True])
def test_matmul_rhs_as_param(lhs_outer, rhs_outer, inner, experimental_bfp8_weights):
    dtype = torch.bfloat16
    matmul = Matmul(inner, rhs_outer, dtype=dtype)
    compiler_config = CompilerConfig(
        experimental_bfp8_weights=experimental_bfp8_weights
    )

    run_op_test_with_random_inputs(
        matmul,
        [(lhs_outer, inner)],
        dtype=dtype,
        framework=Framework.TORCH,
        compiler_config=compiler_config,
    )
