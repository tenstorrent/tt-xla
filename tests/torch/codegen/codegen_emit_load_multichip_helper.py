# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Standalone tensor-parallel (SPMD) driver for test_codegen_emit_load_multichip.

Run in its own process (see the test) so torch_xla's in-process graph cache
doesn't leak between the emit and load phases. Codegen emit vs. load is selected
by the TTXLA_CODEGEN_EXPORT_DIR / TTXLA_CODEGEN_LOAD_DIR env vars the test sets.

Usage: python codegen_emit_load_multichip_helper.py

NOTE: keep this at module scope -- do NOT wrap the device work in a main()
function. The XLA device/tensors must stay alive until interpreter shutdown; as
function locals they are GC'd when the function returns, which deadlocks
torch_xla's TT device teardown and the process hangs on exit. Module globals are
torn down cleanly at finalization.
"""

import os

# Must be set before importing torch_xla so the SHLO->Shardy path is taken.
os.environ["CONVERT_SHLO_TO_SHARDY"] = "1"

import numpy as np
import torch
import torch.nn as nn
import torch_xla
import torch_xla.distributed.spmd as xs
import torch_xla.runtime as xr
from torch_xla.distributed.spmd import Mesh

# Only load mode executes the generated code and returns real tensors; emit mode
# is a dry run that returns zero buffers, so correctness is checked only when
# loading.
check_output = os.environ.get("TTXLA_CODEGEN_LOAD_DIR") is not None

xr.set_device_type("TT")
xr.use_spmd()
device = torch_xla.device()

num_devices = xr.global_runtime_device_count()
mesh = Mesh(np.array(range(num_devices)), (1, num_devices), ("batch", "model"))

torch.manual_seed(0)
model = nn.Sequential(nn.Linear(64, 128), nn.ReLU(), nn.Linear(128, 64)).to(
    torch.bfloat16
)
x = torch.randn(8, 64, dtype=torch.bfloat16)
golden = model(x)

model = model.to(device)
xs.mark_sharding(model[0].weight, mesh, ("model", None))
xs.mark_sharding(model[0].bias, mesh, ("model",))
xs.mark_sharding(model[2].weight, mesh, (None, "model"))
xs.mark_sharding(model[2].bias, mesh, (None,))

x_dev = x.to(device)
xs.mark_sharding(x_dev, mesh, (None, None))

out = model(x_dev)
torch_xla.sync()
out_cpu = out.detach().cpu().float()
if check_output:
    print(f"match: {torch.allclose(out_cpu, golden.float(), atol=0.1)}")
else:
    print("emitted")
