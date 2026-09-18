import os
import numpy as np
import torch
import torch_xla.core.xla_model as xm
import torch_xla.distributed.spmd as xs
import torch_xla.runtime as xr
from torch_xla.distributed.spmd import Mesh

os.environ["CONVERT_SHLO_TO_SHARDY"] = "1"
xr.use_spmd(); xr.set_device_type("TT"); torch.manual_seed(0)
n = xr.global_runtime_device_count()
mesh = Mesh(np.array(range(n)), (1, n), ("batch", "model"))
xs.set_global_mesh(mesh)
xla = xm.xla_device()

V, H, T = 262144, 2816, 256
x = (torch.rand(1, T, V, dtype=torch.float32) * 62.0 - 24.75).to(torch.bfloat16)
W = (torch.randn(V, H, dtype=torch.float32) * 0.02).to(torch.bfloat16)

fn = lambda a, b: torch.matmul(a.softmax(dim=-1, dtype=torch.float32).to(b.dtype), b)

for label, spec in (("sharded", ("model", None)), ("replicated", None)):
    w = W.clone().to(xla)
    if spec:
        xs.mark_sharding(w, mesh, spec)       # shard the CONTRACTING dim
    with torch.no_grad():
        out = torch.compile(fn, backend="tt")(x.clone().to(xla), w)
        xm.mark_step(); out = out.to("cpu").float()
    print(f"{label:>11}: inf={int(out.isinf().sum()):>7}  "
          f"max|out|={float(out[out.isfinite()].abs().max()):.4e}")

print(f"{'CPU':>11}: inf=0        max|out|="
      f"{float(torch.matmul(x.softmax(-1, dtype=torch.float32).to(W.dtype), W).abs().max()):.4e}")
