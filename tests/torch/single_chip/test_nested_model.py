#!/usr/bin/env python3
import os
import torch
import torch.nn as nn

# # Enable our source location feature
# os.environ["XLA_USE_SOURCE_LOCATIONS"] = "1"
os.environ["XLA_HLO_DEBUG"] = "1"

# Enable ABSL logging to see our debug messages
os.environ["ABSL_LOGTOSTDERR"] = "1"
os.environ["ABSL_MINLOGLEVEL"] = "0"  # 0=INFO, 1=WARNING, 2=ERROR

import torch_xla
import torch_xla.core.xla_model as xm
import torch_xla.runtime as xr


class InnerModule(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(5, 3)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.linear(x) # Line 26
        x = self.relu(x) # Line 27
        return x
    


class OuterModule(nn.Module):
    def __init__(self):
        super().__init__()
        self.inner = InnerModule()
        self.final_linear = nn.Linear(3, 1)

    def forward(self, x):
        x = self.inner(x)      # Line 39
        x = torch.relu(x)      # Line 40
        return self.final_linear(x)  # Line 41

def main():
    # Set up XLA runtime for TT backend
    xr.set_device_type("TT")
    
    
    device = xm.xla_device()
    model = OuterModule().to(device)
    x = torch.randn(2, 5).to(device)

    model = torch.compile(model, backend="tt")
    result = model(x)
    
    print(result)

    # # Get XLA text to see our source locations
    # xla_text = torch_xla._XLAC._get_xla_tensors_text([result])
    # print(xla_text)


if __name__ == "__main__":
    main()