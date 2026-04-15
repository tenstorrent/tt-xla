import torch
import torch.nn as nn
import numpy as np
import torch_xla.core.xla_model as xm
import torch_xla.runtime as xr

# 1. Setup Device
xr.set_device_type("TT")
device = xm.xla_device()

# 2. Load Pre-saved Artifacts
# These must exist in your directory from the previous run
x_tt = torch.load("tt_blocks12_out.pt").cpu()   # NPU-noisy input
x_cpu = torch.load("cpu_blocks12_out.pt").cpu() # Pure CPU input
ln_state = torch.load("layer_norm_weights.pt")

# 3. Setup LayerNorm
embed_dim = ln_state["weight"].shape[0]
norm = nn.LayerNorm(embed_dim, eps=1e-6).to(torch.bfloat16) 
norm.load_state_dict(ln_state)
norm.eval()

# 4. CPU Golden Run: norm_cpu(cpu_data)
with torch.no_grad():
    golden_out = norm(x_cpu)

# 5. NPU Real-World Run: norm_npu(npu_data)
# Move norm and data to NPU, compile, and run
norm_npu = norm.to(device)
compiled_norm = torch.compile(norm_npu, backend="tt")

with torch.no_grad():
    npu_out_device = compiled_norm(x_tt.to(device))
    xm.mark_step()
    npu_out = npu_out_device.cpu()

# 6. Calculate PCC
def calculate_pcc(a, b):
    a_f = a.float().numpy().flatten()
    b_f = b.float().numpy().flatten()
    if not (np.isfinite(a_f).all() and np.isfinite(b_f).all()):
        return float("nan")
    return float(np.corrcoef(a_f, b_f)[0, 1])

pcc = calculate_pcc(golden_out, npu_out)

print(f"\n--- Simplified LayerNorm Repro ---")
print(f"Input PCC (TT vs CPU blocks): {calculate_pcc(x_tt, x_cpu):.6f}")
print(f"Output PCC (The smoking gun): {pcc:.6f}")
print(f"----------------------------------")

if pcc < 0.99:
    print("SUCCESS: PCC drop to ~0.64 replicated locally!")
