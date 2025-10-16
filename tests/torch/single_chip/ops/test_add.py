# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
from infra.utilities.types import Framework
from utils import Category

from tests.infra.testers.compiler_config import CompilerConfig
from tests.infra.testers.single_chip.op.op_tester import run_op_test_with_random_inputs


@pytest.mark.push
@pytest.mark.nightly
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
@pytest.mark.parametrize("format", ["float32", "bfloat16", "bfp8"])
def test_add(x_shape: tuple, y_shape: tuple, format: str):
    def add(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return torch.add(x, y)

    if format == "float32":
        dtype = torch.float32
        compiler_config = None
    elif format == "bfloat16":
        dtype = torch.bfloat16
        compiler_config = None
    elif format == "bfp8":
        dtype = torch.bfloat16
        compiler_config = CompilerConfig(enable_bfp8_conversion=True)

    run_op_test_with_random_inputs(
        add,
        [x_shape, y_shape],
        dtype=dtype,
        compiler_config=compiler_config,
        framework=Framework.TORCH,
    )
