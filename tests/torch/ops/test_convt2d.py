import pytest
import torch
from infra import Framework, run_op_test_with_random_inputs
from utils import Category
from loguru import logger

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
                128, 128, kernel_size=(2, 2), stride=(2, 2), bias=False
            )

        def forward(self, input_tensor):
            return self.conv_transpose(input_tensor)

    model = ConvTranspose2dModel()
    model.eval()
    logger.info("model={}",model)
    input_shape = (1, 128, 124, 108)


    run_op_test_with_random_inputs(
        model, [input_shape], framework=Framework.TORCH
    )

