# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
import torch
import torch.nn as nn
import torch_xla.core.xla_model as xm
import torch_xla.runtime as xr
from torch.utils._pytree import tree_map
from tt_torch.backend.backend import xla_backend


class Model(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input_ids, last_hidden_state):

        pooled_output = last_hidden_state[
            torch.arange(last_hidden_state.shape[0], device=last_hidden_state.device),
            input_ids.to(torch.int).argmax(dim=-1).to(last_hidden_state.device),
        ]
        return pooled_output


def test_sanity():

    # ----------------------------------------
    # MODEL
    # ----------------------------------------

    model = Model()

    # ----------------------------------------
    # INPUTS
    # ----------------------------------------

    input_ids = torch.tensor(
        [
            [49406, 320, 1125, 539, 320, 2368, 49407, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [49406, 320, 1125, 539, 320, 1929, 49407, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        ],
        dtype=torch.int64,
    )

    last_hidden_state = torch.rand(2, 16, 512, dtype=torch.bfloat16)

    # ----------------------------------------
    # XLA INFERENCE
    # ----------------------------------------
    xr.set_device_type("TT")

    # Compile the model using XLA
    compiled_model = torch.compile(model, backend=xla_backend)

    # Move model and inputs to the TT device
    device = xm.xla_device()
    compiled_model = compiled_model.to(device)

    def attempt_to_device(x):
        if hasattr(x, "to"):
            return x.to(device)
        return x

    input_ids = tree_map(attempt_to_device, input_ids)
    last_hidden_state = tree_map(attempt_to_device, last_hidden_state)

    # Run inference on Tenstorrent device
    with torch.no_grad():
        output = compiled_model(input_ids, last_hidden_state)
