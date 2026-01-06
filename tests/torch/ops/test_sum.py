import pytest
import torch
from infra import Framework, run_op_test_with_random_inputs
from utils import Category

@pytest.mark.nightly
@pytest.mark.single_device
@pytest.mark.record_test_properties(
    category=Category.OP_TEST,
    torch_op_name="torch.sum",
)

def test_sum_error():

    class Sum(torch.nn.Module):
        def __init__(self,):
            super().__init__()

        def forward(self, x):
            return torch.sum(x).item()

    model = Sum()

    run_op_test_with_random_inputs(
        model, [(1, 3, 1024, 1024)], dtype=torch.bfloat16, framework=Framework.TORCH
    )

@pytest.mark.nightly
@pytest.mark.single_device
@pytest.mark.record_test_properties(
    category=Category.OP_TEST,
    torch_op_name="torch.sum",
)
def test_sum_fix():

    class Sum(torch.nn.Module):
        def __init__(self,):
            super().__init__()

        def forward(self, x):
            return torch.sum(x, dim=(0, 1, 2, 3)).item()

    model = Sum()

    run_op_test_with_random_inputs(
        model, [(1, 3, 1024, 1024)], dtype=torch.bfloat16, framework=Framework.TORCH
    )
    
    