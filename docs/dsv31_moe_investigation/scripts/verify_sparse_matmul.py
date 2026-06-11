"""Offline: recompute fp32 golden for each dumped device sparse_matmul from its
ACTUAL device inputs and PCC vs the device output. Pins whether the sparse_matmul
hardware kernel itself diverges."""
import torch

D = "/localdev/sshon/tt-xla/tmp/op_dump"


def pcc(a, b):
    a = a.detach().to(torch.float32).flatten()
    b = b.detach().to(torch.float32).flatten()
    a = a - a.mean(); b = b - b.mean()
    d = a.norm() * b.norm()
    return (torch.dot(a, b) / d).item() if d > 0 else float("nan")


def load(n):
    return torch.load(f"{D}/{n}.pt", map_location="cpu")


# --- sm0, sm1: gate/up (a=dense [1,AB,M,H], b=weights [1,E,H,N], sp [1,AB,1,E]) ---
for i in (0, 1):
    a = load(f"sm{i}_in0").float()      # [1,128,32,7168]
    w = load(f"sm{i}_in1").float()      # [1,8,7168,2048]
    sp = load(f"sm{i}_in2").float()     # [1,128,1,8]
    out = load(f"sm{i}_out0").float()   # [1,128,1,8,32,2048]
    # golden: per (tile b, expert e): [M,H]@[H,N] = [M,N]
    g = torch.einsum("abmh,aehn->abemn", a, w)        # [1,128,8,32,2048]
    g = g * sp[:, :, 0, :].view(1, 128, 8, 1, 1)       # mask per tile/expert
    g = g.unsqueeze(2)                                  # [1,128,1,8,32,2048]
    print(f"sm{i} (gate/up): out{tuple(out.shape)} golden{tuple(g.shape)} "
          f"PCC={pcc(out, g):.5f}  out_absmax={out.abs().max():.2f} g_absmax={g.abs().max():.2f}")

# --- sm2: down (a=sparse-act [BD?,E,M,K], b=weights [1,E,K,N], sp [1,1,?,E]) ---
a = load("sm2_in0").float()    # [128,8,32,2048]
w = load("sm2_in1").float()    # [1,8,2048,7168]
sp = load("sm2_in2").float()   # [1,1,128,8]
out = load("sm2_out0").float() # [128,8,32,7168]
g = torch.einsum("aemk,ekn->aemn", a, w[0])            # [128,8,32,7168]
g = g * sp[0, 0].view(128, 8, 1, 1)                    # mask
print(f"sm2 (down): out{tuple(out.shape)} golden{tuple(g.shape)} "
      f"PCC={pcc(out, g):.5f}  out_absmax={out.abs().max():.2f} g_absmax={g.abs().max():.2f}")
