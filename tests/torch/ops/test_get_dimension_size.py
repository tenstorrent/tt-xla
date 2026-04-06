# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
import torch_xla
from utils import Category


@pytest.mark.push
@pytest.mark.single_device
@pytest.mark.record_test_properties(
    category=Category.OP_TEST,
    shlo_op_name="stablehlo.get_dimension_size",
)
@pytest.mark.xfail(reason="Dynamic dimensions not supported: result_index=0")
def test_get_dimension_size():
    x_cpu = torch.randn(32, 64)

    # CPU reference: dimension 0 size is 32.
    expected = x_cpu + x_cpu.size(0)

    # TT device execution.
    device = torch_xla.device()
    x = x_cpu.to(device)

    # Mark dimension 0 as dynamic to produce get_dimension_size in the HLO.
    torch_xla._XLAC._xla_mark_dynamic(x, 0)

    dim_size = torch_xla._XLAC._get_xla_tensor_dimension_size(x, 0)
    result = x + dim_size

    # Force compilation and execution on TT device, then compare.
    assert torch.allclose(result.cpu(), expected)
