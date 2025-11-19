# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
Demonstrates how to hook into compile options to use Codegen, from Torch
"""

import os

import torch
import torch.nn as nn
import torch_xla
import torch_xla.core.xla_model as xm
import torch_xla.runtime as xr

# Set up XLA runtime for TT backend.
xr.set_device_type("TT")


class Model(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input_ids, last_hidden_state):

        pooled_output = last_hidden_state[
            torch.arange(last_hidden_state.shape[0], device=last_hidden_state.device),
            input_ids.to(torch.int).argmax(dim=-1).to(last_hidden_state.device),
        ]
        return pooled_output


# Set up compile options to trigger code generation.
options = {
    # Code generation options
    "backend": "codegen_py",
    # Optimizer options
    # "enable_optimizer": True,
    # "enable_memory_layout_analysis": True,
    # "enable_l1_interleaved": False,
    # Tensor dumping options
    # "export_tensors": True,
    "export_path": "owl_vit_emitpy",
}
torch_xla.set_custom_compile_options(options)

# Compile for TT, then move the model and it's inputs to device.
device = xm.xla_device()
model = Model()
model.compile(backend="tt")
model = model.to(device)

input_ids = torch.tensor(
    [
        [49406, 320, 1125, 539, 320, 2368, 49407, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [49406, 320, 1125, 539, 320, 1929, 49407, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    ],
    dtype=torch.int64,
).to(device)

last_hidden_state = torch.rand(2, 16, 512, dtype=torch.bfloat16).to(device)

# Run the model. This triggers code generation.
output = model(input_ids, last_hidden_state)
