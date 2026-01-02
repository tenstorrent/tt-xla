# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
import pytest
import torch
from infra import Framework, run_op_test_with_random_inputs
from loguru import logger
from utils import Category


@pytest.mark.nightly
@pytest.mark.single_device
@pytest.mark.record_test_properties(
    category=Category.OP_TEST,
    torch_op_name="torch.nn.ConvTranspose2d",
)
def test_conv_transpose2d():
    """Test ConvTranspose2d operation."""

    class ConvTranspose2dModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.conv_transpose = torch.nn.ConvTranspose2d(
                2048, 128, kernel_size=(2, 2), stride=(2, 2), bias=False
            )

        def forward(self, input_tensor):
            return self.conv_transpose(input_tensor)

    model = ConvTranspose2dModel()
    model.eval()
    logger.info("model={}", model)
    input_shape = (6, 2048, 8, 22)

    run_op_test_with_random_inputs(model, [input_shape], framework=Framework.TORCH)
