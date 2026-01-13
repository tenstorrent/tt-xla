
import torch.nn as nn
import torch
from infra import Framework, run_op_test_with_random_inputs


def test_this():
    class Slice(nn.Module):
        def __init__(self):
            super().__init__()
            self.W = nn.Parameter(0.02 * torch.randn((32, 1024, 1536),dtype=torch.bfloat16))

        def forward(self, cat_ids):
            return self.W[cat_ids]

    model = Slice()
    cat_ids = torch.ones(1, dtype=torch.int64)
    torch_out = model(cat_ids)
    run_op_test_with_random_inputs(
        Slice(),
        [(1,)],
        dtype = torch.int64,
        framework=Framework.TORCH,
    )