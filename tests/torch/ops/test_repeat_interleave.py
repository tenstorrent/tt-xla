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
    torch_op_name="torch.repeat_interleave",
)
@pytest.mark.parametrize(
    ["input_shape", "repeats"],
    [
        ((128,), 4),
        ((512,), 2),
        ((64,), 8),
    ],
)
def test_repeat_interleave_1d(input_shape: tuple, repeats: int):
    """Test torch.repeat_interleave on 1D tensors — reproduces the GPT-OSS 20B MLP crash.

    In grouped_mm_experts_forward (transformers 5.x moe.py:243), the computation:
        token_idx = torch.arange(num_tokens, device=device).unsqueeze(1).expand(-1, num_top_k).reshape(-1)

    is equivalent to repeat_interleave(arange(num_tokens), num_top_k) and XLA lowers it
    as broadcast_in_dim + reshape. tt-mlir then lowers this to ttnn.repeat_interleave
    on a 1D tensor, which crashes with:
        TT_FATAL: Index is out of bounds for the rank, should be between 0 and 0
                  however is 18446744073709551615
        RuntimeError: Bad StatusOr access: INTERNAL: Error code: 13

    The UINT64_MAX value is a C++ size_t underflow: (size_t)(-1) + 1 = UINT64_MAX,
    triggered in shape.cpp:55 when transpose_impl(-1) is called on a rank-1 tensor.
    """

    class RepeatInterleave1D(torch.nn.Module):
        def __init__(self, repeats):
            super().__init__()
            self.repeats = repeats

        def forward(self, x):
            return torch.repeat_interleave(x, self.repeats)

    run_op_test_with_random_inputs(
        RepeatInterleave1D(repeats),
        [input_shape],
        dtype=torch.float32,
        framework=Framework.TORCH,
    )


@pytest.mark.nightly
@pytest.mark.single_device
@pytest.mark.record_test_properties(
    category=Category.OP_TEST,
    torch_op_name="torch.repeat_interleave",
)
@pytest.mark.parametrize(
    ["num_tokens", "num_top_k"],
    [
        pytest.param(
            128,
            4,
            id="gpt-oss-20b",  # 128 tokens, top-4 routing → token_idx shape (512,)
        ),
        (64, 2),
        (256, 4),
    ],
)
@pytest.mark.xfail(
    reason="arange+unsqueeze+expand+reshape pattern lowers to 1D repeat_interleave in TTNN, "
    "which crashes with TT_FATAL (size_t underflow in transpose_impl for rank-1 tensors). "
    "This is the exact pattern from moe.py:243 in transformers 5.x GPT-OSS 20B MLP."
)
def test_arange_expand_reshape_1d(num_tokens: int, num_top_k: int):
    """Reproduce the exact moe.py:243 pattern that causes the GPT-OSS 20B MLP crash.

    The expression:
        torch.arange(num_tokens).unsqueeze(1).expand(-1, num_top_k).reshape(-1)

    produces a 1D tensor of shape (num_tokens * num_top_k,) where each token index
    is repeated num_top_k times: [0,0,0,0, 1,1,1,1, ..., 127,127,127,127] for 128 tokens × 4.

    XLA traces this as broadcast_in_dim(shape=[num_tokens, num_top_k]) + reshape(shape=[N*K]).
    tt-mlir lowers this broadcast+reshape combination to ttnn.repeat_interleave on the
    intermediate 1D arange tensor, hitting the rank-1 TTNN crash.
    """

    class ArangeExpandReshape(torch.nn.Module):
        def __init__(self, num_tokens, num_top_k):
            super().__init__()
            self.num_tokens = num_tokens
            self.num_top_k = num_top_k

        def forward(self, x):
            # x is a dummy input to satisfy the framework's input requirement;
            # the actual computation only depends on num_tokens and num_top_k.
            token_idx = (
                torch.arange(self.num_tokens, device=x.device)
                .unsqueeze(1)
                .expand(-1, self.num_top_k)
                .reshape(-1)
            )
            # Return as float so PCC comparison works.
            return token_idx.float()

    run_op_test_with_random_inputs(
        ArangeExpandReshape(num_tokens, num_top_k),
        [(num_tokens,)],
        dtype=torch.float32,
        framework=Framework.TORCH,
    )
