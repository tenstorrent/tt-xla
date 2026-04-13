# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Smoke test: compile torch.add(128x128, 128x128) through the TTMetal backend.

Usage:
    python examples/pytorch/d2m/add_smoke.py
"""

import pjrt_plugin_tt

# Select the TTMetal device runtime before any torch_xla import triggers plugin
# registration / device creation. Buffers must be created under the TTMetal
# runtime so their runtime tags match the flatbuffer produced by compilation.
pjrt_plugin_tt.set_backend("ttmetal")

import torch
import torch_xla
import torch_xla.core.xla_model as xm


def main():

    # Route compilation through TTIR → TTMetal → Flatbuffer instead of TTNN.
    torch_xla.set_custom_compile_options(
        {
            "backend": "ttmetal_flatbuffer",
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
