# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
from infra import Framework, run_op_test_with_random_inputs
from utils import Category


@pytest.mark.nightly
@pytest.mark.single_device
@pytest.mark.record_test_properties(
    category=Category.OP_TEST,
    torch_op_name="torch._grouped_mm",
)
@pytest.mark.parametrize(
    ["num_tokens", "in_dim", "out_dim", "num_experts"],
    [
        (64, 256, 256, 4),
        (128, 512, 512, 8),
        (512, 2880, 2880, 32),
    ],
)
def test_grouped_mm(num_tokens: int, in_dim: int, out_dim: int, num_experts: int):
    """Test torch._grouped_mm with the offset (grouped) calling convention.

    aten._grouped_mm(input, weight, offs=offs) multiplies each contiguous group
    of tokens (defined by cumulative offsets) by the corresponding expert weight:

        for expert e: output[offs[e-1]:offs[e]] = input[offs[e-1]:offs[e]] @ weight[e]

    In transformers 5.x MoE models (e.g. GPT-OSS), this is the core computation
    in grouped_mm_experts_forward — the up-projection and down-projection of the
    MLP experts, after tokens have been sorted by expert assignment.

    The weight tensor uses is_transposed=True convention: shape (num_experts, in_dim, out_dim).
    """
    assert num_tokens % num_experts == 0, "num_tokens must be divisible by num_experts"
    tokens_per_expert = num_tokens // num_experts
    offs = torch.arange(1, num_experts + 1, dtype=torch.int32) * tokens_per_expert
    weight = torch.randn(num_experts, in_dim, out_dim, dtype=torch.bfloat16)

    class GroupedMM(torch.nn.Module):
        def __init__(self, weight, offs):
            super().__init__()
            self.register_buffer("weight", weight)
            self.register_buffer("offs", offs)

        def forward(self, x):
            # Cast input to weight dtype (bfloat16), matching the model's behavior.
            return torch._grouped_mm(x.to(self.weight.dtype), self.weight, offs=self.offs)

    run_op_test_with_random_inputs(
        GroupedMM(weight, offs),
        [(num_tokens, in_dim)],
        minval=-1.0,
        maxval=1.0,
        dtype=torch.bfloat16,
        framework=Framework.TORCH,
    )
