# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import torch
import pytest
from infra import run_op_test, Framework
from utils import Category

eltwise_binary_ops = [
    torch.add,
    torch.atan2,
    torch.arctan2,
    torch.bitwise_and,
    torch.bitwise_or,
    torch.bitwise_xor,
    torch.bitwise_left_shift,
    torch.bitwise_right_shift,
    torch.div,
    torch.divide,
    torch.floor_divide,
    torch.fmod,
    torch.logaddexp,
    torch.logaddexp2,
    torch.mul,
    torch.multiply,
    torch.nextafter,
    torch.remainder,
    torch.sub,
    torch.subtract,
    torch.true_divide,
    torch.eq,
    torch.ne,
    torch.le,
    torch.ge,
    torch.greater,
    torch.greater_equal,
    torch.gt,
    torch.less_equal,
    torch.lt,
    torch.less,
    torch.maximum,
    torch.minimum,
    torch.fmax,
    torch.fmin,
    torch.not_equal,
]

# @pytest.mark.push
# @pytest.mark.nightly
@pytest.mark.record_test_properties(category=Category.OP_TEST)
@pytest.mark.parametrize("op", eltwise_binary_ops)
def test_eltwise_binary(op):
    class Binary(torch.nn.Module):
        def forward(self, x, y):
            return op(x, y)

    model = Binary()

    if op in [
        torch.bitwise_and,
        torch.bitwise_or,
        torch.bitwise_xor,
    ]:
        input_x = torch.randint(-100, 100, (32, 32))
        input_y = torch.randint(-100, 100, (32, 32))
        run_op_test(model, [input_x, input_y], framework=Framework.TORCH)

    elif op in [torch.bitwise_left_shift, torch.bitwise_right_shift]:
        # TODO: enable test for these ops once issues is resolved (https://github.com/tenstorrent/tt-torch/issues/1127)
        pytest.skip(f"{op} not supported in tt backend yet. Skipping test.")
    else:
        input_x = torch.randn(32, 32, dtype=torch.bfloat16)
        input_y = torch.randn(32, 32, dtype=torch.bfloat16)
        run_op_test(model, [input_x, input_y], framework=Framework.TORCH)
