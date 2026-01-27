import torch
from infra import Framework, run_op_test_with_random_inputs


class test_cat(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self,x1,x2):
        out = torch.cat([x1, x2], dim=1)
        return out


def test_sanity_1():
    model = test_cat()
    run_op_test_with_random_inputs(
        model,
        [(1,0,1024),(1,0,1024)],
        dtype=torch.float32,
        framework=Framework.TORCH,
    )


def test_sanity_2():
    model = test_cat()
    run_op_test_with_random_inputs(
        model,
        [(1,0,1024),(1,8,1024)],
        dtype=torch.float32,
        framework=Framework.TORCH,
    )