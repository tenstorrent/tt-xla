# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
from infra import Framework, run_graph_test_with_random_inputs
from utils import Category


@pytest.mark.nightly
@pytest.mark.single_device
@pytest.mark.record_test_properties(category=Category.GRAPH_TEST)
@pytest.mark.filecheck(["matmul_permute_a.ttnn.mlir"])
def test_matmul_permute_a(request):
    def matmul_permute_a(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        return torch.matmul(a.permute(0, 2, 1), b)

    run_graph_test_with_random_inputs(
        matmul_permute_a,
        [(2, 64, 32), (2, 64, 16)],
        dtype=torch.bfloat16,
        framework=Framework.TORCH,
        request=request,
    )


@pytest.mark.nightly
@pytest.mark.single_device
@pytest.mark.record_test_properties(category=Category.GRAPH_TEST)
@pytest.mark.filecheck(["matmul_permute_b.ttnn.mlir"])
def test_matmul_permute_b(request):
    def matmul_permute_b(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        return torch.matmul(a, b.permute(0, 2, 1))

    run_graph_test_with_random_inputs(
        matmul_permute_b,
        [(2, 32, 64), (2, 16, 64)],
        dtype=torch.bfloat16,
        framework=Framework.TORCH,
        request=request,
    )


@pytest.mark.nightly
@pytest.mark.single_device
@pytest.mark.record_test_properties(category=Category.GRAPH_TEST)
@pytest.mark.filecheck(["linear_permute_a.ttnn.mlir"])
@pytest.mark.xfail(reason="Turns out linear fusion is broken?")
def test_linear_permute_a(request):
    def linear_permute_a(
        x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor
    ) -> torch.Tensor:
        return torch.nn.functional.linear(x.permute(0, 2, 1), weight, bias)

    run_graph_test_with_random_inputs(
        linear_permute_a,
        [(1, 64, 32), (128, 64), (128,)],
        dtype=torch.bfloat16,
        framework=Framework.TORCH,
        request=request,
    )


@pytest.mark.nightly
@pytest.mark.single_device
@pytest.mark.record_test_properties(category=Category.GRAPH_TEST)
@pytest.mark.filecheck(["linear_permute_b.ttnn.mlir"])
@pytest.mark.xfail(reason="Turns out linear fusion is broken?")
def test_linear_permute_b(request):
    def linear_permute_b(
        x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor
    ) -> torch.Tensor:
        return torch.nn.functional.linear(x, weight.permute(1, 0), bias)

    run_graph_test_with_random_inputs(
        linear_permute_b,
        [(1, 32, 64), (64, 128), (128,)],
        dtype=torch.bfloat16,
        framework=Framework.TORCH,
        request=request,
    )
