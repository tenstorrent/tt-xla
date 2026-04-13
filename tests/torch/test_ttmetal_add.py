# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Smoke test: compile torch.add(128x128, 128x128) through the TTMetal backend.

Usage:
    python tests/torch/test_ttmetal_add.py
"""

import os

# Must be set before any torch_xla import so the PJRT plugin opens the device
# with the TTMetal runtime instead of TTNN.
os.environ["TT_XLA_BACKEND"] = "ttmetal"
# Print IR at every compilation stage to stderr.
os.environ["TTXLA_LOGGER_LEVEL"] = "DEBUG"

import torch
import torch_xla
import torch_xla.core.xla_model as xm


def main():
    # Route compilation through TTIR → TTMetal → Flatbuffer instead of TTNN.
    torch_xla.set_custom_compile_options(
        {
            "backend": "ttmetal_flatbuffer",
            "export_path": "/tmp/ttmetal_add_irs",
        }
    )

    device = xm.xla_device()

    a = torch.randn(128, 128, dtype=torch.bfloat16)
    b = torch.randn(128, 128, dtype=torch.bfloat16)

    golden = torch.add(a, b)

    a_dev = a.to(device)
    b_dev = b.to(device)

    out_dev = torch.add(a_dev, b_dev)

    # Trigger compilation + execution by moving result back to CPU.
    out = out_dev.to("cpu")

    max_diff = (out - golden).abs().max().item()
    print(f"Max absolute difference: {max_diff:.6f}")
    assert max_diff < 0.05, f"Result mismatch: max diff {max_diff}"
    print("PASS")


if __name__ == "__main__":
    main()
