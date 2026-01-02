import pytest
import torch
from infra import Framework, run_op_test_with_random_inputs
from utils import Category
from loguru import logger
import torch

@pytest.mark.nightly
@pytest.mark.single_device
@pytest.mark.record_test_properties(
    category=Category.OP_TEST,
    torch_op_name="torch.nn.MaxPool2d",
)
def test_max_pool2d():
    """Test max_pool2d operation."""

    class MaxPool2d(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.maxpool = torch.nn.MaxPool2d(kernel_size=3, stride=2, padding=1, dilation=1, ceil_mode=False)

        def forward(self, input_tensor):
            return self.maxpool(input_tensor)

    model = MaxPool2d().to(torch.bfloat16)
    
    logger.info("model={}",model)

    input_shape = (1, 64, 608 , 784)

    run_op_test_with_random_inputs(
        model, [input_shape], framework=Framework.TORCH, dtype=torch.bfloat16
    )