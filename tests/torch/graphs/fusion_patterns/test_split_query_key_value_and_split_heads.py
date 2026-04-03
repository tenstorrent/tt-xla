# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
from einops import rearrange
from infra import Framework, run_graph_test_with_random_inputs
from utils import Category

from tests.infra.testers.compiler_config import CompilerConfig


@pytest.mark.extended
@pytest.mark.nightly
@pytest.mark.single_device
@pytest.mark.record_test_properties(category=Category.GRAPH_TEST)
@pytest.mark.filecheck(["split_query_key_value_and_split_heads.ttnn.mlir"])
def test_split_query_key_value_and_split_heads_mha_matmul(request):
    """MHA pattern: equal Q/K/V weight shapes, matmul variant."""
    batch, seq_len, hidden_dim = 1, 32, 512
    num_heads, head_size = 8, 64

    def mha_qkv_matmul(
        x: torch.Tensor,
        wq: torch.Tensor,
        wk: torch.Tensor,
        wv: torch.Tensor,
    ) -> tuple:
        x_2d = rearrange(x, "b s d -> (b s) d")
        q = rearrange(
            torch.matmul(x_2d, wq), "(b s) (h d) -> b h s d", b=batch, h=num_heads
        )
        k = rearrange(
            torch.matmul(x_2d, wk), "(b s) (h d) -> b h s d", b=batch, h=num_heads
        )
        v = rearrange(
            torch.matmul(x_2d, wv), "(b s) (h d) -> b h s d", b=batch, h=num_heads
        )
        return q, k, v

    run_graph_test_with_random_inputs(
        mha_qkv_matmul,
        [
            (batch, seq_len, hidden_dim),
            (hidden_dim, num_heads * head_size),
            (hidden_dim, num_heads * head_size),
            (hidden_dim, num_heads * head_size),
        ],
        dtype=torch.bfloat16,
        framework=Framework.TORCH,
        compiler_config=CompilerConfig(optimization_level=1),
        request=request,
    )


@pytest.mark.extended
@pytest.mark.nightly
@pytest.mark.single_device
@pytest.mark.record_test_properties(category=Category.GRAPH_TEST)
@pytest.mark.filecheck(["split_query_key_value_and_split_heads.ttnn.mlir"])
def test_split_query_key_value_and_split_heads_mha_matmul_with_bias(request):
    """MHA pattern: equal Q/K/V weight shapes, matmul + bias addition variant."""
    batch, seq_len, hidden_dim = 1, 32, 512
    num_heads, head_size = 8, 64

    def mha_qkv_matmul_with_bias(
        x: torch.Tensor,
        wq: torch.Tensor,
        wk: torch.Tensor,
        wv: torch.Tensor,
        bq: torch.Tensor,
        bk: torch.Tensor,
        bv: torch.Tensor,
    ) -> tuple:
        x_2d = rearrange(x, "b s d -> (b s) d")
        q = rearrange(
            torch.matmul(x_2d, wq) + bq, "(b s) (h d) -> b h s d", b=batch, h=num_heads
        )
        k = rearrange(
            torch.matmul(x_2d, wk) + bk, "(b s) (h d) -> b h s d", b=batch, h=num_heads
        )
        v = rearrange(
            torch.matmul(x_2d, wv) + bv, "(b s) (h d) -> b h s d", b=batch, h=num_heads
        )
        return q, k, v

    run_graph_test_with_random_inputs(
        mha_qkv_matmul_with_bias,
        [
            (batch, seq_len, hidden_dim),
            (hidden_dim, num_heads * head_size),
            (hidden_dim, num_heads * head_size),
            (hidden_dim, num_heads * head_size),
            (num_heads * head_size,),
            (num_heads * head_size,),
            (num_heads * head_size,),
        ],
        dtype=torch.bfloat16,
        framework=Framework.TORCH,
        compiler_config=CompilerConfig(optimization_level=1),
        request=request,
    )


@pytest.mark.extended
@pytest.mark.nightly
@pytest.mark.single_device
@pytest.mark.record_test_properties(category=Category.GRAPH_TEST)
@pytest.mark.filecheck(["split_query_key_value_and_split_heads.ttnn.mlir"])
@pytest.mark.xfail(reason="This seems to not trigger the pattern?")
def test_split_query_key_value_and_split_heads_mha_linear(request):
    """MHA pattern: equal Q/K/V weight shapes, F.linear (with bias) variant."""
    batch, seq_len, hidden_dim = 1, 32, 512
    num_heads, head_size = 8, 64

    def mha_qkv_linear(
        x: torch.Tensor,
        wq: torch.Tensor,
        wk: torch.Tensor,
        wv: torch.Tensor,
        bq: torch.Tensor,
        bk: torch.Tensor,
        bv: torch.Tensor,
    ) -> tuple:
        x_2d = rearrange(x, "b s d -> (b s) d")
        q = rearrange(
            torch.nn.functional.linear(x_2d, wq, bq),
            "(b s) (h d) -> b h s d",
            b=batch,
            h=num_heads,
        )
        k = rearrange(
            torch.nn.functional.linear(x_2d, wk, bk),
            "(b s) (h d) -> b h s d",
            b=batch,
            h=num_heads,
        )
        v = rearrange(
            torch.nn.functional.linear(x_2d, wv, bv),
            "(b s) (h d) -> b h s d",
            b=batch,
            h=num_heads,
        )
        return q, k, v

    run_graph_test_with_random_inputs(
        mha_qkv_linear,
        [
            (batch, seq_len, hidden_dim),
            (num_heads * head_size, hidden_dim),
            (num_heads * head_size, hidden_dim),
            (num_heads * head_size, hidden_dim),
            (num_heads * head_size,),
            (num_heads * head_size,),
            (num_heads * head_size,),
        ],
        dtype=torch.bfloat16,
        framework=Framework.TORCH,
        compiler_config=CompilerConfig(optimization_level=1),
        request=request,
    )


@pytest.mark.extended
@pytest.mark.nightly
@pytest.mark.single_device
@pytest.mark.record_test_properties(category=Category.GRAPH_TEST)
@pytest.mark.filecheck(["split_query_key_value_and_split_heads.ttnn.mlir"])
def test_split_query_key_value_and_split_heads_mha_transposed_key(request):
    """MHA pattern: transposed key variant (key permute is [0, 2, 3, 1])."""
    batch, seq_len, hidden_dim = 1, 32, 512
    num_heads, head_size = 8, 64

    def mha_qkv_transposed_key(
        x: torch.Tensor,
        wq: torch.Tensor,
        wk: torch.Tensor,
        wv: torch.Tensor,
    ) -> tuple:
        x_2d = rearrange(x, "b s d -> (b s) d")
        q = rearrange(
            torch.matmul(x_2d, wq), "(b s) (h d) -> b h s d", b=batch, h=num_heads
        )
        k = rearrange(
            torch.matmul(x_2d, wk), "(b s) (h d) -> b h d s", b=batch, h=num_heads
        )
        v = rearrange(
            torch.matmul(x_2d, wv), "(b s) (h d) -> b h s d", b=batch, h=num_heads
        )
        return q, k, v

    run_graph_test_with_random_inputs(
        mha_qkv_transposed_key,
        [
            (batch, seq_len, hidden_dim),
            (hidden_dim, num_heads * head_size),
            (hidden_dim, num_heads * head_size),
            (hidden_dim, num_heads * head_size),
        ],
        dtype=torch.bfloat16,
        framework=Framework.TORCH,
        compiler_config=CompilerConfig(optimization_level=1),
        request=request,
    )


@pytest.mark.extended
@pytest.mark.nightly
@pytest.mark.single_device
@pytest.mark.record_test_properties(category=Category.GRAPH_TEST)
@pytest.mark.filecheck(["split_query_key_value_and_split_heads.ttnn.mlir"])
def test_split_query_key_value_and_split_heads_gqa_matmul(request):
    """GQA pattern: query has more heads than key/value, matmul variant."""
    batch, seq_len, hidden_dim = 1, 32, 512
    num_query_heads, num_kv_heads, head_size = 8, 2, 64

    def gqa_qkv_matmul(
        x: torch.Tensor,
        wq: torch.Tensor,
        wk: torch.Tensor,
        wv: torch.Tensor,
    ) -> tuple:
        x_2d = rearrange(x, "b s d -> (b s) d")
        q = rearrange(
            torch.matmul(x_2d, wq), "(b s) (h d) -> b h s d", b=batch, h=num_query_heads
        )
        k = rearrange(
            torch.matmul(x_2d, wk), "(b s) (h d) -> b h s d", b=batch, h=num_kv_heads
        )
        v = rearrange(
            torch.matmul(x_2d, wv), "(b s) (h d) -> b h s d", b=batch, h=num_kv_heads
        )
        return q, k, v

    run_graph_test_with_random_inputs(
        gqa_qkv_matmul,
        [
            (batch, seq_len, hidden_dim),
            (hidden_dim, num_query_heads * head_size),
            (hidden_dim, num_kv_heads * head_size),
            (hidden_dim, num_kv_heads * head_size),
        ],
        dtype=torch.bfloat16,
        framework=Framework.TORCH,
        compiler_config=CompilerConfig(optimization_level=1),
        request=request,
    )
