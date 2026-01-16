# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
Example: Export IR files during compilation.

Demonstrates:
  - Multiple graphs per model (g0=forward, g1=backward)
  - Graph counter resets when export_model_name changes

Output: ./ir_export/irs/*.mlir
"""

import torch
import torch.nn as nn
import torch_xla
import torch_xla.runtime as xr


class SimpleMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(64, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        return self.fc2(torch.relu(self.fc1(x)))


def run_model(name: str, batch_size: int, device):
    """Run forward + backward, generating g0 and g1."""
    torch_xla.set_custom_compile_options(
        {
            "export_path": "ir_export",
            "export_model_name": f"{name}_bs{batch_size}",
        }
    )

    model = SimpleMLP().to(torch.bfloat16).to(device)
    x = torch.randn(batch_size, 64, dtype=torch.bfloat16, device=device)

    # Forward (g0) + Backward (g1)
    out = model(x)
    torch_xla.sync()
    out.sum().backward()
    torch_xla.sync()

    print(f"Compiled: {name}_bs{batch_size} (g0: forward, g1: backward)")


def main():
    xr.set_device_type("TT")
    device = torch_xla.device()

    print("Exporting IR files for each compilation...\n")

    run_model("mlp", batch_size=4, device=device)
    run_model("mlp", batch_size=8, device=device)

    print("\nExported IR files to: ir_export/irs/")
    print("  Stages: vhlo, shlo, ttir, ttnn")
    print("  Pattern: <stage>_<model_name>_g<N>_<timestamp>.mlir")


if __name__ == "__main__":
    main()
