# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""
Reproducer for extremely slow consteval when both matmul operands are registered buffers.

When a model has two registered buffers used as matmul inputs, insert_argument_type_markers
marks them as type="constant". tt-mlir then attempts to consteval matmul(c1, c2) at
compile time. For large 3D shapes, consteval completes but is extremely slow — execution
time grows rapidly with output matrix size (small: ~3s, medium: ~1m30s, large: ~8m30s,
extra_large: ~26m).

Slowdown is observed during XLA lazy sync (tensor materialization), not in Python-level
torch.compile infrastructure.

Related: consteval performance with 3D constant tensor as matmul input.
Reported from tt-forge-sweeps matmul MP tests (CONST_EVAL_PASS scenario).
"""

import pytest
import torch
import torch.nn as nn
from infra import Framework, run_graph_test
from tests.infra.testers.compiler_config import CompilerConfig


class ConstEvalMatmul(nn.Module):
    """
    Model with two registered buffers used as matmul operands.
    Simulates CONST_EVAL_PASS scenario from tt-forge-sweeps matmul MP tests.

    forward computes:
        mm1 = matmul(c1, c2)   # both are constants -> consteval candidate
        mm2 = matmul(x, y)     # user inputs -> runtime
        return add(mm1, mm2)
    """

    def __init__(self, shape_c1, shape_c2):
        super().__init__()
        self.register_buffer("c1", torch.randn(*shape_c1, dtype=torch.bfloat16))
        self.register_buffer("c2", torch.randn(*shape_c2, dtype=torch.bfloat16))

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        mm1 = torch.matmul(self.c1, self.c2)
        mm2 = torch.matmul(x, y)
        return torch.add(mm1, mm2)


@pytest.mark.push
@pytest.mark.single_device
@pytest.mark.parametrize(
    ["shape_c1", "shape_c2"],
    [
        # Small shape - passes (~3s)
        [(1, 32, 256), (256, 256)],
        # Medium shape - passes slowly (~1m 30s)
        [(32, 128, 2304), (2304, 2048)],
        # Large shape - passes very slowly (~8m 30s)
        [(32, 128, 3072), (3072, 8192)],
        # Extra large shape - passes very slowly (~26m)
        [(32, 128, 3072), (3072, 24576)],
    ],
    ids=["small", "medium", "large", "extra_large"],
)
def test_matmul_const_eval_slow(shape_c1, shape_c2):
    """
    Reproducer: matmul with both operands as registered buffers (constants).
    Consteval is extremely slow for large 3D shapes — all cases pass locally
    but execution time grows rapidly with output matrix size.
    """
    model = ConstEvalMatmul(shape_c1, shape_c2)

    x = torch.randn(*shape_c1)
    y = torch.randn(*shape_c2)

    run_graph_test(
        model,
        [x, y],
        framework=Framework.TORCH,
        compiler_config=CompilerConfig(
            optimization_level=0,
            fp32_dest_acc_en=True,
            math_fidelity="hifi4",
        ),
    )
