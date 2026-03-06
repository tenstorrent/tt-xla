# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
from infra import Framework, run_graph_test_with_random_inputs
from utils import Category

from tests.infra.testers.compiler_config import CompilerConfig


@pytest.mark.push
@pytest.mark.nightly
@pytest.mark.single_device
@pytest.mark.record_test_properties(category=Category.GRAPH_TEST)
@pytest.mark.filecheck(["sdpa.ttnn.mlir"])
def test_sdpa(request):
    def sdpa(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        scale = q.shape[-1] ** -0.5
        mask = torch.triu(
            torch.full(
                (1, 1, q.shape[-2], q.shape[-2]),
                float("-inf"),
                dtype=q.dtype,
                device=q.device,
            ),
            diagonal=1,
        )
        scores = torch.matmul(q, k.transpose(-2, -1)) * scale + mask
        return torch.matmul(torch.nn.functional.softmax(scores, dim=-1), v)

    run_graph_test_with_random_inputs(
        sdpa,
        [(1, 8, 32, 64), (1, 8, 32, 64), (1, 8, 32, 64)],
        dtype=torch.bfloat16,
        framework=Framework.TORCH,
        compiler_config=CompilerConfig(optimization_level=1),
        request=request,
    )
