# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import math
import torch
import torch.nn as nn

from ...utils import TensorShape
from ...utils import TensorUtils


class ModelFromAnotherOp(nn.Module):
    def __init__(self, operator, kwargs):
        super().__init__()
        self.testname = "Element_wise_unary_operators_test_op_src_from_another_op"
        self.operator = operator
        self.kwargs = kwargs

    def forward(self, x):
        xx = torch.add(x, x)
        return self.operator(xx, **self.kwargs)


class ModelDirect(nn.Module):
    def __init__(self, operator, kwargs):
        super().__init__()
        self.testname = "Element_wise_unary_operators_test_op_src_from_host"
        self.operator = operator
        self.kwargs = kwargs

    def forward(self, x):
        return self.operator(x, **self.kwargs)


class ModelConstEvalPass(nn.Module):
    def __init__(self, operator, shape: TensorShape, kwargs, dtype, value_range):
        super().__init__()
        self.testname = "Element_wise_unary_operators_test_op_src_const_eval_pass"
        self.operator = operator
        self.kwargs = kwargs
        self.c = TensorUtils.create_torch_constant(
            input_shape=shape,
            dev_data_format=dtype,
            value_range=value_range,
            random_seed=math.prod(shape),
        )
        self.register_buffer("constant", self.c)

    def forward(self, x):
        cc = self.operator(self.c, **self.kwargs)
        xx = self.operator(x, **self.kwargs)
        return torch.add(xx, cc)
