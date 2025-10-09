# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
from infra import Framework, run_op_test_with_random_inputs
from utils import Category


@pytest.mark.push
@pytest.mark.nightly
@pytest.mark.record_test_properties(category=Category.OP_TEST)
def test_silu():
    class Silu(torch.nn.Module):
        def forward(self, x):
            return torch.nn.functional.silu(x)

    silu = Silu()

    run_op_test_with_random_inputs(
        silu, [(32, 32)], dtype=torch.bfloat16, framework=Framework.TORCH
    )


@pytest.mark.push
@pytest.mark.nightly
@pytest.mark.record_test_properties(category=Category.OP_TEST)
def test_silu_with_dtype_promotion():
    class Silu(torch.nn.Module):
        def forward(self, x):
            res = torch.nn.functional.silu(x)
            return res.to(torch.float32)

    silu = Silu()

    run_op_test_with_random_inputs(
        silu, [(32, 32)], dtype=torch.float32, framework=Framework.TORCH
    )
