# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
Demonstrates how to hook into compile options to use Codegen, from Torch
"""

import shutil
from pathlib import Path

import torch
import torch.nn as nn
import torch_xla.runtime as xr
from tt_torch import codegen_py

class MaxPool2d(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.maxpool = torch.nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

    def forward(self, input_tensor):
        return self.maxpool(input_tensor)


def main():
    # Set up XLA runtime for TT backend.
    xr.set_device_type("TT")

    # Any compile options you could specify when executing the model normally can also be used with codegen.
    extra_options = {
        # "optimization_level": 0,  # Levels 0, 1, and 2 are supported
    }
    
    model = MaxPool2d()
    model.eval()
    x = torch.randn(6, 64, 464, 800)

    codegen_py(model, x, export_path="maxpool2d", compiler_options=extra_options, export_tensors=False)


if __name__ == "__main__":
    main()
