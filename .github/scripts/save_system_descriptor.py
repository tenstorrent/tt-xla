# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
Initializes a TT device and saves the system descriptor to disk.
Used by CI to capture the hardware system descriptor as an artifact
for use in compile-only tests.

Usage:
    python save_system_descriptor.py <output_prefix>

    e.g. python save_system_descriptor.py system_descriptor/tt_system
         -> creates system_descriptor/tt_system_system_desc.ttsys
"""

import sys

import torch
import torch_xla.core.xla_model as xm
import torch_xla.runtime as xr
from ttxla_tools.serialization import save_system_descriptor_to_disk


def main():
    output_prefix = sys.argv[1] if len(sys.argv) > 1 else "system_descriptor/tt_system"

    xr.set_device_type("TT")
    device = xm.xla_device()

    x = torch.randn(2, 2).to(device)
    xm.mark_step()

    save_system_descriptor_to_disk(output_prefix)
    print(f"System descriptor saved to {output_prefix}_system_desc.ttsys")


if __name__ == "__main__":
    main()
