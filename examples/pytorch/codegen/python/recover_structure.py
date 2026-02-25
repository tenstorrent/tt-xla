# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
Demonstrates how to hook into compile options to use Codegen, from Torch
"""

import os
import shutil
from pathlib import Path

import torch
import torch.nn as nn
import torch_xla
import torch_xla.core.xla_model as xm
import torch_xla.runtime as xr
from tt_torch import codegen_py


class FirstModule(nn.Module):
    def __init__(self):
        super().__init__()
        self.w = nn.Parameter(torch.randn(32, 64, dtype=torch.bfloat16))

    def forward(self, x):
        return torch.matmul(x, self.w)


class SecondModule(nn.Module):
    def __init__(self):
        super().__init__()
        self.w = nn.Parameter(torch.randn(64, 128, dtype=torch.bfloat16))

    def forward(self, x):
        return torch.matmul(x, self.w)


class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.m1 = FirstModule()
        self.m2 = SecondModule()

    def forward(self, x):
        x = self.m1(x)
        x = self.m2(x)
        # return torch.sum(x**2)  # problem with simple/ast trace when using pow
        # return torch.sum(x)  # add another op outside of first/second modules
        # ^ torch.sum seems to be ignored, doesn't appear in the IR
        return x * x  # add another op outside of first/second modules


def main():
    """Run codegen with structure recovery."""
    # Enable HLO debug output
    os.environ["XLA_HLO_DEBUG"] = "1"

    # Set up XLA runtime for TT backend.
    xr.set_device_type("TT")

    # Any compile options you could specify when executing the model normally can also be used with codegen.
    extra_options = {
        "codegen_try_recover_structure": True,  # experimental feature
        "export_tensors": True,
    }

    model = Model()
    x = torch.randn(32, 32, dtype=torch.bfloat16)

    codegen_py(
        model,
        x,
        export_path="recover_structure_example",
        compiler_options=extra_options,
    )


def test_recover_structure():
    """Test that codegen with structure recovery creates the expected output folder."""
    output_dir = Path("recover_structure_example")
    if output_dir.exists():
        shutil.rmtree(output_dir)
    try:
        main()
        assert (
            output_dir.exists()
        ), f"Expected output folder '{output_dir}' was not created"
        assert output_dir.is_dir(), f"'{output_dir}' exists but is not a directory"
    finally:
        if output_dir.exists():
            shutil.rmtree(output_dir)


if __name__ == "__main__":
    main()
