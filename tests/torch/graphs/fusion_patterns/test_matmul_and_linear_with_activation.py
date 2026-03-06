# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
import torch.nn.functional as F
from infra import Framework, run_graph_test_with_random_inputs
from utils import Category


@pytest.mark.nightly
@pytest.mark.single_device
@pytest.mark.record_test_properties(category=Category.GRAPH_TEST)
@pytest.mark.filecheck(["matmul_with_activation_sigmoid.ttnn.mlir"])
def test_matmul_with_sigmoid(request):
    def matmul_sigmoid(x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(torch.matmul(x, w))

    run_graph_test_with_random_inputs(
        matmul_sigmoid,
        [(1, 32, 64), (64, 128)],
        dtype=torch.bfloat16,
        framework=Framework.TORCH,
        request=request,
    )


@pytest.mark.push
@pytest.mark.nightly
@pytest.mark.single_device
@pytest.mark.record_test_properties(category=Category.GRAPH_TEST)
@pytest.mark.filecheck(["matmul_with_activation_silu.ttnn.mlir"])
def test_matmul_with_silu(request):
    def matmul_silu(x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
        return F.silu(torch.matmul(x, w))

    run_graph_test_with_random_inputs(
        matmul_silu,
        [(1, 32, 64), (64, 128)],
        dtype=torch.bfloat16,
        framework=Framework.TORCH,
        request=request,
    )


@pytest.mark.nightly
@pytest.mark.single_device
@pytest.mark.record_test_properties(category=Category.GRAPH_TEST)
@pytest.mark.filecheck(["matmul_with_activation_gelu.ttnn.mlir"])
def test_matmul_with_gelu(request):
    def matmul_gelu(x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
        return F.gelu(torch.matmul(x, w))

    run_graph_test_with_random_inputs(
        matmul_gelu,
        [(1, 32, 64), (64, 128)],
        dtype=torch.bfloat16,
        framework=Framework.TORCH,
        request=request,
    )
