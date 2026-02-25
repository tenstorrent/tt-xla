# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import itertools

import pytest
import torch
import torch_xla.core.xla_model as xm
from infra.evaluators import ComparisonConfig, PccConfig, TorchComparisonEvaluator
from utils import Category

# Test entirely vibecoded


class InplaceMutationModel(torch.nn.Module):
    def __init__(
        self, inplace_first: bool, inplace_last: bool, only_inplace: bool = False
    ):
        super().__init__()
        self.inplace_first = inplace_first
        self.inplace_last = inplace_last
        self.only_inplace = only_inplace

    def forward(self, x):
        if self.only_inplace:
            x.add_(1.0)
            return x
        if self.inplace_first:
            x.add_(1.0)
        y = x * 2.0
        if self.inplace_last:
            y.add_(3.0)
        return y


def run_reference(
    input_cpu, pre_mutate, inplace_first, inplace_last, only_inplace=False
):
    model = InplaceMutationModel(inplace_first, inplace_last, only_inplace=only_inplace)
    x = input_cpu.clone()
    if pre_mutate:
        x.add_(1.0)
    with torch.no_grad():
        y = model(x)
    return y, x


def run_compiled(
    input_cpu, pre_mutate, inplace_first, inplace_last, only_inplace=False
):
    model = InplaceMutationModel(inplace_first, inplace_last, only_inplace=only_inplace)
    compiled_model = torch.compile(model, backend="tt")
    device = xm.xla_device()
    compiled_model = compiled_model.to(device)
    x = input_cpu.clone().to(device)
    if pre_mutate:
        x.add_(1.0)
    y = compiled_model(x)
    return y, x


@pytest.mark.push
@pytest.mark.nightly
@pytest.mark.single_device
@pytest.mark.record_test_properties(category=Category.GRAPH_TEST)
@pytest.mark.parametrize(
    "pre_mutate,inplace_first,inplace_last",
    list(itertools.product([False, True], repeat=3)),
)
def test_inplace_mutations_with_intermediate_ops(
    pre_mutate, inplace_first, inplace_last
):
    torch.manual_seed(0)
    input_cpu = torch.randn(8, 8, dtype=torch.float32)

    expected_output, expected_input = run_reference(
        input_cpu, pre_mutate, inplace_first, inplace_last
    )
    device_output, device_input = run_compiled(
        input_cpu, pre_mutate, inplace_first, inplace_last
    )

    comparison_config = ComparisonConfig(pcc=PccConfig(required_pcc=0.9999))
    comparator = TorchComparisonEvaluator(comparison_config)
    comparator.evaluate(device_output.cpu(), expected_output)
    comparator.evaluate(device_input.cpu(), expected_input)


@pytest.mark.push
@pytest.mark.nightly
@pytest.mark.single_device
@pytest.mark.record_test_properties(category=Category.GRAPH_TEST)
@pytest.mark.parametrize("pre_mutate", [False, True])
def test_inplace_only_graph(pre_mutate):
    torch.manual_seed(0)
    input_cpu = torch.randn(8, 8, dtype=torch.float32)

    expected_output, expected_input = run_reference(
        input_cpu,
        pre_mutate,
        inplace_first=False,
        inplace_last=False,
        only_inplace=True,
    )
    device_output, device_input = run_compiled(
        input_cpu,
        pre_mutate,
        inplace_first=False,
        inplace_last=False,
        only_inplace=True,
    )

    comparison_config = ComparisonConfig(pcc=PccConfig(required_pcc=0.9999))
    comparator = TorchComparisonEvaluator(comparison_config)
    comparator.evaluate(device_output.cpu(), expected_output)
    comparator.evaluate(device_input.cpu(), expected_input)
