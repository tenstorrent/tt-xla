# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch


@pytest.mark.nightly
@pytest.mark.single_device
def test_tensor_attributes_torch_override():
    """
    Test that torch tensor attributes are accessed correctly when override is active.
    """
    from tt_torch.torch_overrides import torch_function_override

    def test_ordering(input_ids):
        shape_0 = input_ids.shape[0]
        device = input_ids.device
        return device

    input_ids = torch.tensor([[1, 2, 3], [4, 5, 6]], dtype=torch.int32)

    compiled_op = torch.compile(test_ordering)
    output = compiled_op(input_ids)

    assert output == torch.device("cpu"), f"Expected device to be cpu, but got {output}"
