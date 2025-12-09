# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
import torch.nn.functional as F
from infra import Framework, run_op_test_with_random_inputs
from infra.comparators.torch_comparator import TorchComparator
from utils import Category

from tests.infra.comparators.comparison_config import ComparisonConfig
from tests.infra.testers.compiler_config import CompilerConfig


class Linear(torch.nn.Module):
    def __init__(self, in_features, out_features, bias=True, dtype=torch.float32):
        super().__init__()
        self.linear = torch.nn.Linear(in_features, out_features, bias=bias, dtype=dtype)

    def forward(self, x):
        return self.linear(x)


@pytest.mark.push
@pytest.mark.nightly
@pytest.mark.single_device
@pytest.mark.record_test_properties(category=Category.OP_TEST)
@pytest.mark.parametrize("batch_size", [32, 64])
@pytest.mark.parametrize("in_features", [32, 64])
@pytest.mark.parametrize("out_features", [32, 64])
@pytest.mark.parametrize("bias", [False, True])
@pytest.mark.parametrize("experimental_enable_weight_bfp8_conversion", [False, True])
def test_linear(
    batch_size,
    in_features,
    out_features,
    bias,
    experimental_enable_weight_bfp8_conversion,
):
    dtype = torch.bfloat16
    linear = Linear(in_features, out_features, bias=bias, dtype=dtype)
    compiler_config = CompilerConfig(
        experimental_enable_weight_bfp8_conversion=experimental_enable_weight_bfp8_conversion
    )

    run_op_test_with_random_inputs(
        linear,
        [(batch_size, in_features)],
        dtype=dtype,
        framework=Framework.TORCH,
        compiler_config=compiler_config,
    )


@pytest.mark.nightly
@pytest.mark.single_device
@pytest.mark.record_test_properties(category=Category.OP_TEST)
def test_linear_torch_override():
    """
    Test that TorchFunctionOverride produces correct results for linear operations with
    4D input tensors by comparing override output against standard torch.nn.functional.linear.

    This tests the override in eager mode (without torch.compile) since the override
    has `not torch.compiler.is_compiling()` check.
    """
    from tt_torch.torch_overrides import torch_function_override

    dtype = torch.bfloat16
    in_features = 96
    out_features = 384

    input_tensor = torch.randn(1, 128, 128, in_features, dtype=dtype)
    weight = torch.randn(out_features, in_features, dtype=dtype)
    bias = torch.randn(out_features, dtype=dtype)

    # Temporarily disable the override to get golden output
    torch_function_override.__exit__(None, None, None)
    try:
        golden = F.linear(input_tensor, weight, bias)
    finally:
        torch_function_override.__enter__()  # Always re-enable it

    # Compute actual output with override active (eager mode, not compiled)
    output = F.linear(input_tensor, weight, bias)

    comparator = TorchComparator(ComparisonConfig())
    comparator.compare(output, golden)
