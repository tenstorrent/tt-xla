"""Test if tt::sampling also hangs when it's the only op in a compiled graph."""
import os
os.environ["TT_USE_TTNN_SAMPLING"] = "1"

import torch
import torch_xla
import torch_xla.core.xla_model as xm
import sys
import time

device = xm.xla_device()

@torch.compile(backend="tt", fullgraph=True, dynamic=False)
def sample_only(values, indices, k, p, temp):
    return torch.ops.tt.sampling(values, indices, k, p, temp, seed=42)

# Minimal inputs matching ttnn.sampling requirements
batch = 32
candidates = 32
values = torch.randn(batch, candidates, dtype=torch.bfloat16).to(device)
indices = torch.arange(candidates).unsqueeze(0).expand(batch, -1).to(torch.int32).to(device)
k = torch.full((batch,), candidates, dtype=torch.int32).to(device)
p = torch.ones(batch, dtype=torch.bfloat16).to(device)
temp = torch.full((batch,), 1.667, dtype=torch.bfloat16).to(device)

print("Compiling tt::sampling standalone...", flush=True)
t0 = time.perf_counter()
result = sample_only(values, indices, k, p, temp)
torch_xla.sync()
t1 = time.perf_counter()
print(f"Done in {t1-t0:.2f}s", flush=True)
print(f"Result: {result.cpu()}", flush=True)
print("PASSED", flush=True)
sys.stdout.flush()
os._exit(0)
