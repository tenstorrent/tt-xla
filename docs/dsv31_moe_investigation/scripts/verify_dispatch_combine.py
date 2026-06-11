"""Verify the device all_to_all_combine (and dispatch) kernels against their CPU
reference impls, using the device's ACTUAL dumped inputs (shard 0)."""
import torch
import tt_torch.custom_ops  # registers torch.ops.tt.*

D = "/localdev/sshon/tt-xla/tmp/op_dump"


def pcc(a, b):
    a = a.detach().to(torch.float32).flatten()
    b = b.detach().to(torch.float32).flatten()
    a = a - a.mean(); b = b - b.mean()
    d = a.norm() * b.norm()
    return (torch.dot(a, b) / d).item() if d > 0 else float("nan")


def load(n):
    return torch.load(f"{D}/{n}.pt", map_location="cpu")


# ===== COMBINE: in0=expert outs [8,1,4096,7168], in1=metadata [1,1,4096,8],
#                in2=mapping [1,1,256,32]; out [8,1,1024,7168] =====
in0 = load("comb0_in0")
meta = load("comb0_in1").to(torch.int64)
mapping = load("comb0_in2").to(torch.int64)
dev_out = load("comb0_out0")
cpu_out = torch.ops.tt.all_to_all_combine(
    in0, meta, mapping, num_devices=4, cluster_axis=0,
    num_experts_per_tok=8, output_shard_dim=2,
)
print(f"COMBINE: dev{tuple(dev_out.shape)} cpu{tuple(cpu_out.shape)} "
      f"PCC={pcc(dev_out, cpu_out):.5f}  "
      f"dev_absmax={dev_out.abs().max():.3f} cpu_absmax={cpu_out.abs().max():.3f} "
      f"dev_nonzero={(dev_out!=0).float().mean():.3f} cpu_nonzero={(cpu_out!=0).float().mean():.3f}")

# ===== DISPATCH: in0=hidden [64,1,16,7168], in1=indices [64,1,16,8],
#                 in2=mapping [1,1,256,32]; out0 dispatched [1,256,16,7168] =====
h = load("disp0_in0")
idx = load("disp0_in1").to(torch.int64)
mapping_d = load("disp0_in2").to(torch.int64)
dev_disp = load("disp0_out0")
cpu_disp, cpu_meta = torch.ops.tt.all_to_all_dispatch(
    h, idx, mapping_d, num_devices=4, cluster_axis=0,
)
print(f"DISPATCH: dev{tuple(dev_disp.shape)} cpu{tuple(cpu_disp.shape)} "
      f"PCC={pcc(dev_disp, cpu_disp):.5f}  "
      f"dev_absmax={dev_disp.abs().max():.3f} cpu_absmax={cpu_disp.abs().max():.3f} "
      f"dev_nonzero={(dev_disp!=0).float().mean():.3f} cpu_nonzero={(cpu_disp!=0).float().mean():.3f}")
