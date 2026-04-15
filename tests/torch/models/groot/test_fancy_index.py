# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
Minimal reproducer for ttnn::embedding L1 overflow via fancy indexing.

No model dependencies — just nn.Parameter + tensor[index].

The GR00T model's CategorySpecificLinear does self.W[cat_ids] which the
compiler lowers to ttnn::embedding. layer2's W=[32, 1024, 1536] overflows
L1, while layer1's W=[32, 64, 1024] passes.

This file tests all the W/b shapes from the model to prove the op fails
purely based on tensor size, independent of the model.
"""

import torch
import pytest
from infra import Framework, run_op_test
from utils import Category


class FancyIndexWrapper(torch.nn.Module):
    """Wraps a single nn.Parameter + fancy index: param[index]."""

    def __init__(self, shape, dtype=torch.bfloat16):
        super().__init__()
        self.param = torch.nn.Parameter(0.02 * torch.randn(*shape, dtype=dtype))

    def forward(self, index):
        return self.param[index]


# All W and b shapes from CategorySpecificLinear in the GR00T action head.
# cat_ids = embodiment_id = tensor([0]) — single index lookup.

PARAM_SHAPES = {
    "state_encoder_layer1_W":   (32, 64, 1024),     # ~4 MB
    "state_encoder_layer1_b":   (32, 1024),          # ~64 KB
    "state_encoder_layer2_W":   (32, 1024, 1536),    # ~96 MB  ← FAILS
    "state_encoder_layer2_b":   (32, 1536),          # ~96 KB
    "action_decoder_layer1_W":  (32, 1024, 1024),    # ~64 MB
    "action_decoder_layer1_b":  (32, 1024),          # ~64 KB
    "action_decoder_layer2_W":  (32, 1024, 32),      # ~2 MB
    "action_decoder_layer2_b":  (32, 32),            # ~2 KB
    "action_encoder_W1":        (32, 32, 1536),      # ~3 MB
    "action_encoder_W2":        (32, 3072, 1536),    # ~288 MB
    "action_encoder_W3":        (32, 1536, 1536),    # ~144 MB
}


@pytest.mark.parametrize("name,shape", list(PARAM_SHAPES.items()), ids=list(PARAM_SHAPES.keys()))
@pytest.mark.single_device
@pytest.mark.record_test_properties(category=Category.OP_TEST)
def test_fancy_index(name, shape):
    """
    param[index] where param has the given shape and index=tensor([0]).
    Compiler lowers this to ttnn::embedding.
    """
    size_bytes = 1
    for d in shape:
        size_bytes *= d
    size_bytes *= 2

    print(f"\n[{name}] shape={list(shape)}, size={size_bytes:,} bytes ({size_bytes / 1024 / 1024:.1f} MB)")

    index = torch.tensor([0], dtype=torch.long)

    wrapper = FancyIndexWrapper(shape)
    wrapper.eval()

    run_op_test(
        wrapper, [index],
        framework=Framework.TORCH,
    )
