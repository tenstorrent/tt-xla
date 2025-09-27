# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import torch
import torch_xla.core.xla_model as xm

import pytest
from infra.utilities.types import Framework
import tt_torch_backend.custom_ops
import tt_torch_backend
from tests.infra.testers.single_chip.op.op_tester import OpTester, run_op_test

# TODO: Record superset properties for these tests.


@pytest.mark.parametrize("num_heads", [12, 16])
@pytest.mark.parametrize("max_seq_len", [64, 128])
@pytest.mark.parametrize("head_size", [64, 128])
@pytest.mark.parametrize("seq_len_to_fill", [32, 64])
def test_fill_cache(num_heads, max_seq_len, head_size, seq_len_to_fill):

    cache = torch.zeros(1, num_heads, max_seq_len, head_size, dtype=torch.bfloat16)
    fill_value = torch.randn(
        1, num_heads, seq_len_to_fill, head_size, dtype=torch.bfloat16
    )

    run_op_test(
        torch.ops.tt.fill_cache, [cache, fill_value, 0], framework=Framework.TORCH
    )


@pytest.mark.parametrize("num_heads", [12, 16])
@pytest.mark.parametrize("max_seq_len", [64, 128])
@pytest.mark.parametrize("head_size", [64, 128])
def test_update_cache(num_heads, max_seq_len, head_size):

    cache = torch.zeros(1, num_heads, max_seq_len, head_size, dtype=torch.bfloat16)
    fill_value = torch.randn(1, num_heads, 1, head_size, dtype=torch.bfloat16)

    cache_position = torch.tensor([10], dtype=torch.int32)

    run_op_test(
        torch.ops.tt.update_cache,
        [cache, fill_value, cache_position, 0],
        framework=Framework.TORCH,
    )
