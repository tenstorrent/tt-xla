# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
from infra import Framework, run_op_test_with_random_inputs
from utils import Category

from tests.infra.testers.compiler_config import CompilerConfig


class Linear(torch.nn.Module):
    def __init__(self, in_features, out_features, bias=True, dtype=torch.float32):
        super().__init__()
        self.linear = torch.nn.Linear(in_features, out_features, bias=bias, dtype=dtype)

    def forward(self, x):
        return self.linear(x)


@pytest.mark.push
@pytest.mark.nightly
@pytest.mark.record_test_properties(category=Category.OP_TEST)
@pytest.mark.parametrize("batch_size", [32, 64])
@pytest.mark.parametrize("in_features", [32, 64])
@pytest.mark.parametrize("out_features", [32, 64])
@pytest.mark.parametrize("bias", [False, True])
@pytest.mark.parametrize("experimental_bfp8_weights", [False, True])
def test_linear(batch_size, in_features, out_features, bias, experimental_bfp8_weights):
    dtype = torch.bfloat16
    linear = Linear(in_features, out_features, bias=bias, dtype=dtype)
    compiler_config = CompilerConfig(
        experimental_bfp8_weights=experimental_bfp8_weights
    )

    run_op_test_with_random_inputs(
        linear,
        [(batch_size, in_features)],
        dtype=dtype,
        framework=Framework.TORCH,
        compiler_config=compiler_config,
    )
