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
@pytest.mark.xfail(reason="Dynamic dimensions not supported")
def test_get_dimension_size():
    device = torch_xla.device()
    x = torch.randn(32, 64).to(device)

    # Mark dimension 0 as dynamic to produce get_dimension_size in the HLO.
    torch_xla._XLAC._xla_mark_dynamic(x, 0)

    def get_dim_size(x: torch.Tensor) -> torch.Tensor:
        dim_size = torch_xla._XLAC._get_xla_tensor_dimension_size(x, 0)
        return x + dim_size

    result = get_dim_size(x)

    # Force compilation and execution on TT device.
    result.cpu()
