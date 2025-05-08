# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
import torch
import torch_xla.core.xla_model as xm
import torch_xla.runtime as xr
import torch_xla.distributed.spmd as xs
from torch_xla.distributed.spmd import Mesh

import os

# Set SPMD mode with auto-sharding enabled
xr.use_spmd(auto=True)

os.environ["PJRT_DEVICE"] = "TT"
os.environ["XLA_STABLEHLO_COMPILE"] = "1"

from torch_xla.experimental import plugins


class TTPjrtPlugin(plugins.DevicePlugin):
    def library_path(self):
        return os.path.join(
            os.path.dirname(__file__), "../build/src/tt/pjrt_plugin_tt.so"
        )


plugins.register_plugin("TT", TTPjrtPlugin())

device = xm.xla_device()

t1 = torch.randn(800, 400)
t2 = torch.randn(400, 800)

# Calculate golden matmul result on CPU
golden = t1 @ t2

# Move tensors to XLA device (aka TT chips).
# Torch-XLA should automatically shard the tensors across the devices
# without any `mark_sharding` hints.
t1 = t1.to(device)
t2 = t2.to(device)

# Calculate the matmul result on XLA device
test_result = t1 @ t2

# Move the result back to CPU for comparison
test_result = test_result.to("cpu")

# Check if the result is close to the golden result
if torch.allclose(test_result, golden):
    print("Test passed: The results are close.")
else:
    print("Test failed: The results are not close.")
    print(f"Test result: {test_result}")
    print(f"Golden result: {golden}")
