# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pytest
import torch
from infra import Framework, run_op_test
from infra.comparators.torch_comparator import TorchComparator
from utils import Category

from tests.infra.comparators.comparison_config import ComparisonConfig


@pytest.mark.nightly
@pytest.mark.single_device
@pytest.mark.record_test_properties(category=Category.OP_TEST)
def test_numpy_scalar_division_torch_override():
    """
    Test that TorchFunctionOverride works correctly for numpy scalar division operations.
    """
    from tt_torch.torch_overrides import torch_function_override

    def test_div(w):
        return w / 2 / 8

    w = np.float32(16.0)

    # Temporarily disable the override to get golden output
    torch_function_override.__exit__(None, None, None)
    try:
        golden_compiled_op = torch.compile(test_div)
        golden = golden_compiled_op(w)
        golden = torch.tensor(golden)
    finally:
        torch_function_override.__enter__()  # Always re-enable it

    compiled_op = torch.compile(test_div)
    output = compiled_op(w)
    output = torch.tensor(output)

    comparator = TorchComparator(ComparisonConfig())
    comparator.compare(output, golden)
