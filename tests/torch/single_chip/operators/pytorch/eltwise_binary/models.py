# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

# Model for binary operators


import math
import torch

from ...utils import TensorUtils


class ModelFromAnotherOp(torch.nn.Module):

    model_name = "model_op_src_from_another_op"

    def __init__(self, operator, opname, shape, kwargs):
        super(ModelFromAnotherOp, self).__init__()
        self.testname = "pytorch_eltwise_binary_" + opname + "_model_from_another_op"
        self.operator = operator
        self.kwargs = kwargs

    def forward(self, x: torch.Tensor, y: torch.Tensor):
        # we use Add and Subtract operators to create two operands which are inputs for the binary operator
        xx = torch.add(x, y)
        yy = torch.add(x, y)  # TODO temporary we use add operator, return to sub later
        output = self.operator(xx, yy, **self.kwargs)
        return output


class ModelDirect(torch.nn.Module):

    model_name = "model_op_src_from_host"

    def __init__(self, operator, opname, shape, kwargs):
        super(ModelDirect, self).__init__()
        self.testname = "pytorch_eltwise_binary_" + opname + "_model_direct"
        self.operator = operator
        self.kwargs = kwargs

    def forward(self, x: torch.Tensor, y: torch.Tensor):
        output = self.operator(x, y, **self.kwargs)
        return output


class ModelConstEvalPass(torch.nn.Module):

    model_name = "model_op_src_const_eval_pass"

    def __init__(self, operator, opname, shape, kwargs, dtype, value_range):
        super(ModelConstEvalPass, self).__init__()
        self.testname = "pytorch_eltwise_binary_" + opname + "_model_const_eval_pass"
        self.operator = operator
        self.kwargs = kwargs

        self.c1 = TensorUtils.create_torch_constant(
            input_shape=shape,
            dev_data_format=dtype,
            value_range=value_range,
            random_seed=math.prod(shape),
        )
        self.c2 = TensorUtils.create_torch_constant(
            input_shape=shape,
            dev_data_format=dtype,
            value_range=value_range,
            random_seed=sum(shape),
        )
        self.register_buffer("constant1", self.c1)
        self.register_buffer("constant2", self.c2)

    def forward(self, x, y):
        v1 = self.operator(self.c1, self.c2, **self.kwargs)
        # v2 and v3 consume inputs
        v2 = torch.add(x, y)
        v3 = torch.add(v1, v2)
        return v3
