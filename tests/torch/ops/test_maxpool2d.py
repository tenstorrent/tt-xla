import pytest
import torch
from infra import Framework, run_op_test_with_random_inputs
from utils import Category
from loguru import logger


@pytest.mark.parametrize(
    "input_shape, kernel_size, stride, padding, dilation, ceil_mode",
    [
        ((6, 64, 256, 704), 3, 2, 1, 1, False), # R50dcn_Gridmask_C5, R50dcn_Gridmask_P4
        ((6, 256, 80, 200), 3, 2, 0, 1, True), # Vovnet_Gridmask_P4_800x320
        ((6, 256, 160, 400), 3, 2, 0, 1, True), # Vovnet_Gridmask_P4_1600x640
    ],
)
def test_max_pool2d(input_shape, kernel_size, stride, padding, dilation, ceil_mode):
    """Test max_pool2d operation."""

    class MaxPool2d(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.pool = torch.nn.MaxPool2d(
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                dilation=dilation,
                ceil_mode=ceil_mode,
            )

        def forward(self, input_tensor):
            return self.pool(input_tensor)

    model = MaxPool2d()
    model.eval()

    logger.info("model={}", model)
    logger.info("input_shape={}", input_shape)

    run_op_test_with_random_inputs(
        model, [input_shape], framework=Framework.TORCH
    )
