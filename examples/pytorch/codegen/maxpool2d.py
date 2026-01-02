import torch
import torch_xla.runtime as xr
from loguru import logger

from tt_torch import codegen_py

# Set up XLA runtime for TT backend
xr.set_device_type("TT")

class MaxPool2d(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.maxpool = torch.nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)

        def forward(self, input_tensor):
            return self.maxpool(input_tensor)

model = MaxPool2d().to(torch.bfloat16)

logger.info("model={}",model)

x= torch.randn(1, 64, 480 , 640, dtype=torch.bfloat16)

codegen_py(model,x , export_path="maxpool2d_carvana_emitpy",export_tensors=False)