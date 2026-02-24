# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
from infra import Framework, run_op_test
from utils import Category


@pytest.mark.push
@pytest.mark.nightly
@pytest.mark.single_device
@pytest.mark.record_test_properties(
    category=Category.OP_TEST,
    torch_op_name="torch.view_as_complex",
)
@pytest.mark.parametrize(
    "shape",
    [(32, 32), (64, 64)],
    ids=lambda val: f"{val}",
)
def test_complex(shape: tuple):
    class Complex(torch.nn.Module):
        def forward(self, real: torch.Tensor, imag: torch.Tensor) -> torch.Tensor:
            return torch.view_as_complex(torch.stack([real, imag], dim=-1))

    real = torch.randn(shape, dtype=torch.float32)
    imag = torch.randn(shape, dtype=torch.float32)

    run_op_test(Complex(), [real, imag], framework=Framework.TORCH)
