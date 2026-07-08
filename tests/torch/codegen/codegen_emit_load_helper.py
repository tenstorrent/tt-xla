# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Standalone MLP driver for test_codegen_emit_load.

Run in its own process (see the test) so torch_xla's in-process graph cache
doesn't leak between the emit and load phases. Codegen emit vs. load is selected
by the TTXLA_CODEGEN_EXPORT_DIR / TTXLA_CODEGEN_LOAD_DIR env vars the test sets.

Runs num_graphs forwards with distinct batch sizes so each compiles a separate
graph.

Usage: python codegen_emit_load_helper.py <num_graphs>

NOTE: keep this at module scope -- do NOT wrap the device work in a main()
function. The XLA device/tensors must stay alive until interpreter shutdown; as
function locals they are GC'd when the function returns, which deadlocks
torch_xla's TT device teardown and the process hangs on exit. Module globals are
torn down cleanly at finalization.
"""

import os
import sys

import torch
import torch.nn as nn
import torch_xla
import torch_xla.runtime as xr

num_graphs = int(sys.argv[1])

# Only load mode executes the generated code and returns real tensors; emit mode
# is a dry run that returns zero buffers, so correctness is checked only when
# loading.
check_output = os.environ.get("TTXLA_CODEGEN_LOAD_DIR") is not None

xr.set_device_type("TT")
device = torch_xla.device()

torch.manual_seed(0)
model = nn.Sequential(nn.Linear(64, 128), nn.ReLU(), nn.Linear(128, 10)).to(
    torch.bfloat16
)
inputs = [torch.randn(4 * 2**i, 64, dtype=torch.bfloat16) for i in range(num_graphs)]
goldens = [model(x) for x in inputs]

xla_model = model.to(device)
for x, golden in zip(inputs, goldens):
    out = xla_model(x.to(device))
    torch_xla.sync()
    if check_output:
        ok = torch.allclose(out.cpu().float(), golden.float(), atol=0.1)
        print(f"batch {x.shape[0]} match: {ok}")
        assert ok
    else:
        print(f"batch {x.shape[0]} emitted")
