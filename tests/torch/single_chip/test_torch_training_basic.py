# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
import torch
import pytest
from torch._dynamo.backends.common import aot_autograd
from torch._dynamo.backends.registry import lookup_backend
from torch.utils._pytree import tree_map, tree_flatten
import copy
from infra.comparators.torch_comparator import TorchComparator
from tests.infra.comparators.comparison_config import (
    AtolConfig,
    ComparisonConfig,
)

# Binding model and loss together to compile them in a single graph, so that
# tt-torch backend can capture the backward pass as well.
class BindModelAndLoss(torch.nn.Module):
    def __init__(self, model, loss_fn):
        super().__init__()
        self.model = model
        self.loss_fn = loss_fn

    def forward(self, x):  # could add y for supervised loss
        pred = self.model(x)
        loss = self.loss_fn(pred)  # self.loss_fn(pred, y)
        return pred, loss


def test_bw_compile():
    class l1_loss(torch.nn.Module):
        def forward(self, a):
            return a.mean()

    class Linear(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = torch.nn.Linear(10, 10)

        def forward(self, a):
            x = self.linear(a)
            return x

    # torch.set_default_device("xla")
    input = torch.randn(10, 10, device="cpu", requires_grad=True)
    input_xla = input.detach().to("xla").requires_grad_(True)
    model = Linear()
    model_cpu = copy.deepcopy(model)
    tt_backend = lookup_backend("tt")
    model_and_loss = BindModelAndLoss(model, l1_loss()).to("xla")
    mod = torch.compile(
        model_and_loss,
        backend=aot_autograd(fw_compiler=tt_backend, bw_compiler=tt_backend),
    )
    result_xla, loss = mod(input_xla)
    # NOTE: Printing the loss here can materialize a separate subgraph for loss.
    # print(f'loss : {loss}')
    loss.backward()

    input_cpu = input
    result_cpu = model_cpu(input_cpu)
    loss_cpu = l1_loss()(result_cpu)
    loss_cpu.backward()

    comparator = TorchComparator(
        ComparisonConfig(
            atol=AtolConfig(enabled=True, required_atol=0.02),
        )
    )
    # Compare the gradients from backward pass
    comparator.compare(model.linear.weight.grad, model_cpu.linear.weight.grad)


if __name__ == "__main__":
    test_bw_compile()
