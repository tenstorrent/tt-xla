# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
Save a system descriptor from live hardware to disk.

Producing a descriptor needs hardware (it reads a live chip); compiling against
one does not. Run this on the target arch's hardware to regenerate the checked-in
`<arch>_system_desc.ttsys` used by test_compile_only.py:

    python save_system_desc.py system_descs/<arch>
"""

import sys

import torch_xla.core.xla_model as xm
import torch_xla.runtime as xr
from ttxla_tools import save_system_descriptor_to_disk


def save_system_desc(output_prefix: str):
    """Save the system descriptor from live hardware to disk."""
    xr.set_device_type("TT")
    xm.xla_device()
    save_system_descriptor_to_disk(output_prefix)


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print(f"Usage: {sys.argv[0]} <output_prefix>")
        sys.exit(1)
    save_system_desc(sys.argv[1])