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
    torch_op_name="torch.argsort",
)
@pytest.mark.parametrize(
    "input_shape",
    [
        (1, 128),
        (1, 512),
        (4, 32),
    ],
)
def test_argsort_2d(input_shape: tuple):
    """Test torch.argsort on 2D tensors (expected to work on TT hardware)."""

    class Argsort(torch.nn.Module):
        def forward(self, x):
            return torch.argsort(x)

    run_op_test_with_random_inputs(
        Argsort(),
        [input_shape],
        dtype=torch.float32,
        framework=Framework.TORCH,
    )


@pytest.mark.nightly
@pytest.mark.single_device
@pytest.mark.record_test_properties(
    category=Category.OP_TEST,
    torch_op_name="torch.argsort",
)
@pytest.mark.parametrize(
    "input_shape",
    [
        (128,),
        pytest.param(
            (512,),
            id="gpt-oss-20b",  # 128 tokens x top-4 routing = 512 expert assignments
        ),
        (256,),
    ],
)
@pytest.mark.xfail(
    reason="1D argsort crashes TTNN: sort calls squeeze_from_4D which asserts rank >= 2. "
    "In GPT-OSS 20B, perm = torch.argsort(expert_ids) on shape (512,) triggers this crash. "
    "Fix: decompose argsort on 1D tensors or unsqueeze before sort. "
    "See: https://github.com/tenstorrent/tt-xla/issues/XXXX"
)
def test_argsort_1d(input_shape: tuple):
    """Test torch.argsort on 1D tensors — reproduces the GPT-OSS 20B crash.

    In grouped_mm_experts_forward (transformers 5.x), argsort is called as:
        perm = torch.argsort(expert_ids)  # expert_ids shape: (num_tokens * top_k,) = (512,)

    The 1D sort triggers TTNN's squeeze_from_4D which asserts rank >= 2, causing:
        TT_FATAL: Can't convert shape rank 1 to 4D
        RuntimeError: Bad StatusOr access: INTERNAL: Error code: 13

    Note: "While executing %histc" in the model error log is misleading — histc just
    triggers the XLA lazy graph sync; argsort accumulated earlier is what crashes.
    """

    class Argsort1D(torch.nn.Module):
        def forward(self, x):
            return torch.argsort(x)

    run_op_test_with_random_inputs(
        Argsort1D(),
        [input_shape],
        dtype=torch.float32,
        framework=Framework.TORCH,
    )
