# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
Test to verify that explorer build for op by op tests has tool capability.
"""

import chisel
import golden
import torch
import torch_xla
import torch_xla.runtime as xr
import ttmlir
from _ttmlir_runtime.runtime import DebugHooks, unregister_hooks

xr.set_device_type("TT")


def preop(a, b, c):
    print("Callback")


a = torch.ones((128, 128)).to("xla")
b = torch.rand((128, 128)).to("xla")

hooks = DebugHooks.get(pre_op=preop)
print(hooks)

a = torch.ones((128, 128)).to("xla")
b = torch.rand((128, 128)).to("xla")
print(a * 2 + b)
torch_xla.sync(wait=True)

unregister_hooks()
