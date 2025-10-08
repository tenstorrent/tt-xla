import os
import torch
import torch_xla
import torch_xla.runtime as xr
import torch_xla.core.xla_model as xm

def test_add():
    class AddModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
        def forward(self, x):
            return x+1

    xr.set_device_type("TT")
    device = torch_xla.device()

    x:torch.Tensor = torch.ones((3,3)).to(device)
    model = AddModel()
    model = model.to(device)
    model.compile(backend='tt')
    y = model(x)

    print(y)
