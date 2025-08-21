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

from torch.fx import GraphModule, Node
from tt_torch.tools.utils import CompilerConfig

# Purpose:
#   Compose an arbitrary model and an arbitrary loss function into a single
#   callable so BOTH the model’s forward/backward and the loss’s forward/backward
#   can be captured by torch.compile / AOTAutograd / your custom backend—without
#   modifying the original model.
#
# Intent:
#   - Keep model and loss independent & easily swappable.
#   - Return (y, loss) so callers can still inspect/use the raw model output
#     while driving gradients via loss.backward().
#   - Assumes loss_fn takes only the model output: loss_fn(y).
#     For supervised losses that require targets, extend forward(...) to accept
#     a target and pass it through (e.g., loss_fn(y, target)).
#
# Usage:
#   compiled = torch.compile(ModelThenLoss(model, loss_fn), backend=..., fullgraph=True)
#   y, loss = compiled(x)
#   loss.backward()
#
# Notes:
#   - Ensure device/dtype match across x, model, and loss_fn (e.g., all on "xla").
#   - If your pipeline prefers returning only the loss, you can change the return
#     signature to just `return loss` (or gate it behind a flag).
class ModelThenLoss(torch.nn.Module):
    def __init__(self, model, loss_fn):
        super().__init__()
        self.model = model
        self.loss_fn = loss_fn

    def forward(self, x):
        y = self.model(x)
        loss = self.loss_fn(y)
        return y, loss


def test_bw_compile():
    class l1_loss(torch.nn.Module):
        def forward(self, a):
            return a.mean()

    class Linear(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = torch.nn.Linear(10, 100)

        def forward(self, a):
            x = self.linear(a)
            return x

    input = torch.randn(10, 10, device="cpu", requires_grad=True)
    input_xla = input.detach().to("xla").requires_grad_(True)
    model = Linear()
    model_cpu = copy.deepcopy(model)
    model_loss = ModelThenLoss(model, l1_loss())
    tt_backend = lookup_backend("tt")
    # torch._dynamo.config.compiled_autograd = False

    # Configure forward and backward separately
    fw_config = CompilerConfig()
    fw_config.push_outputs_to_cpu = False
    fw_config.is_backward = False
    bw_config = CompilerConfig()
    bw_config.is_backward = True
    bw_config.push_outputs_to_cpu = False

    tt_fw = lambda gm, ex: tt_backend(gm, ex, options=fw_config)
    tt_bw = lambda gm, ex: tt_backend(gm, ex, options=bw_config)

    mod = torch.compile(
        model_loss.to("xla"),
        backend=aot_autograd(fw_compiler=tt_fw, bw_compiler=tt_bw),
        fullgraph=True,
    )

    # return would be result, loss
    _, loss = mod(input_xla)
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
