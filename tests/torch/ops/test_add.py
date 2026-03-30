# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
from infra.utilities.types import Framework
from utils import Category

from tests.infra.testers.single_chip.op.op_tester import run_op_test_with_random_inputs

# NOTE: This test passes `request` to support serialization (--serialize).
# Other op tests can follow this pattern. See docs/src/test_infra.md for details.


@pytest.mark.push
@pytest.mark.nightly
@pytest.mark.single_device
@pytest.mark.record_test_properties(
    category=Category.OP_TEST,
    torch_op_name="torch.add",
)
@pytest.mark.parametrize(
    ["x_shape", "y_shape"],
    [
        [(32, 32), (32, 32)],
        [(64, 64), (64, 64)],
    ],
    ids=lambda val: f"{val}",
)
@pytest.mark.parametrize("format", ["float32", "bfloat16"])
def test_add(x_shape: tuple, y_shape: tuple, format: str, request):
    def add(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return torch.add(x, y)

    if format == "float32":
        dtype = torch.float32
    else:
        dtype = torch.bfloat16

    run_op_test_with_random_inputs(
        add,
        [x_shape, y_shape],
        dtype=dtype,
        framework=Framework.TORCH,
        request=request,
    )
