# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import os

import numpy as np
import torch
import torch.nn as nn
import torch_xla
import torch_xla.distributed.spmd as xs
import torch_xla.runtime as xr
from torch_xla.distributed.spmd import Mesh


class LayerNormModel(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.linear = nn.Linear(hidden_size, hidden_size)
        self.ln = nn.LayerNorm(hidden_size)

    def forward(self, x):
        x = self.linear(x)
        x = self.ln(x)
        return x


def layernorm_sharded():
    # Enable SPMD mode.
    os.environ["CONVERT_SHLO_TO_SHARDY"] = "1"
    xr.use_spmd()

    # Create a (1, 2) mesh with "batch" and "model" axes.
    num_devices = xr.global_runtime_device_count()
    assert num_devices == 2, f"Expected 2 devices, got {num_devices}"
    device_ids = np.array(range(num_devices))
    mesh = Mesh(device_ids, (1, 2), ("batch", "model"))

    hidden_size = 128
    model = LayerNormModel(hidden_size).to(torch.bfloat16)

    # Compute golden reference on CPU before moving to device.
    input_cpu = torch.randn(1, 32, hidden_size, dtype=torch.bfloat16)
    with torch.no_grad():
        expected = model(input_cpu)

    # Move model and input to device.
    device = torch_xla.device()
    model = model.to(device)

    input_tensor = input_cpu.to(device)

    # Shard the activation on the "model" axis (last dimension).
    # The linear weight is column-parallel so the output is sharded on dim=-1.
    xs.mark_sharding(model.linear.weight, mesh, ("model", None))  # column-parallel
    xs.mark_sharding(input_tensor, mesh, (None, None, "model"))  # shard hidden dim

    # Compile and run.
    compiled_model = torch.compile(model, backend="tt")

    with torch.no_grad():
        output = compiled_model(input_tensor)

    output_cpu = output.cpu()

    # Validate against CPU golden reference.
    print("Output shape:", output_cpu.shape)
    print("Expected (first 5):", expected[0, 0, :5])
    print("Got      (first 5):", output_cpu[0, 0, :5])

    if torch.allclose(output_cpu, expected, atol=1e-1, rtol=1e-1):
        print("PASS: TT output matches CPU reference.")
    else:
        max_diff = (output_cpu - expected).abs().max().item()
        print(f"FAIL: Max absolute difference: {max_diff}")


if __name__ == "__main__":
    xr.set_device_type("TT")
    layernorm_sharded()
