# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
from infra.utilities.types import Framework
from utils import Category

from tests.infra.testers.single_chip.op.op_tester import run_op_test_with_random_inputs


@pytest.mark.nightly
@pytest.mark.single_device
@pytest.mark.record_test_properties(
    category=Category.OP_TEST,
    torch_op_name="torch.linalg.inv",
)
def test_linalg_inv():

    def linalg_inv(x: torch.Tensor) -> torch.Tensor:
        return torch.linalg.inv(x)

    input_shape = (3, 3)

    run_op_test_with_random_inputs(
        linalg_inv,
        [input_shape],
        dtype=torch.float32,
        framework=Framework.TORCH,
    )