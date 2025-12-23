import torch
import torch_xla.runtime as xr
from tt_torch import codegen_py

# Set up XLA runtime for TT backend
xr.set_device_type("TT")



class GridSample(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input_tensor, grid):
        return torch.nn.functional.grid_sample(
            input_tensor,
            grid,
        )

model = GridSample()

input = torch.randn(1, 256, 28, 28)
grid = torch.randn(1, 7, 25281, 2)

x = [input,grid]

codegen_py(model, *x, export_path="grid_sample_oft_emitpy", export_tensors=False)