"""TRUE distributed all_to_all_combine golden vs device dump (nopad, 32 dev, 8 exp/dev).
For device d, original token t (first tpd=1024 slots), expert-slot k:
  g = metadata[d][t,k] (global expert id); if g//8==d: out=down_out[d][g-8d, t] else 0.
"""
import torch
D = "tmp/op_dump"
L = lambda p: torch.load(f"{D}/{p}.pt", map_location="cpu")
def pcc(a, b):
    a=a.detach().float().flatten(); b=b.detach().float().flatten()
    a=a-a.mean(); b=b-b.mean(); d=a.norm()*b.norm()
    return (torch.dot(a,b)/d).item() if d>0 else float("nan")

meta = L("comb0_in1_all").long()        # [32,1,1,4096,8]
down = L("comb0_in0_all")               # [32,8,1,4096,7168] bf16
devout = L("comb0_out0_all")            # [32,8,1,1024,7168]
ND, ELOC = down.shape[0], down.shape[1]
TPD = devout.shape[3]                    # 1024
H = down.shape[-1]; K = devout.shape[1]
print(f"ND={ND} E_local={ELOC} TPD={TPD} K={K} H={H}")

tidx = torch.arange(TPD)
pccs_true = []; pccs_cpuimpl = []
nz_dev=[]; nz_true=[]
for d in range(ND):
    meta_d = meta[d,0,0,:TPD,:]          # [TPD, K] global ids
    down_d = down[d,:,0,:TPD,:].float()  # [E_local, TPD, H]
    dev_d  = devout[d,:,0,:,:].float()   # [K, TPD, H]
    golden = torch.zeros(K, TPD, H)
    for k in range(K):
        g = meta_d[:,k]
        valid = (g // ELOC) == d
        local = (g - d*ELOC).clamp(0, ELOC-1)
        gathered = down_d[local, tidx, :]          # [TPD, H]
        golden[k] = gathered * valid.unsqueeze(-1)
    pccs_true.append(pcc(dev_d, golden))
    nz_dev.append((dev_d!=0).float().mean().item())
    nz_true.append((golden!=0).float().mean().item())
    if d < 4:
        print(f"dev{d}: PCC(true)={pccs_true[-1]:.5f} nz_dev={nz_dev[-1]:.4f} nz_true={nz_true[-1]:.4f}")
import statistics as st
print(f"TRUE combine golden: mean PCC over {ND} devices = {st.mean(pccs_true):.5f} (min {min(pccs_true):.4f} max {max(pccs_true):.4f})")
print(f"mean nz dev={st.mean(nz_dev):.4f} true={st.mean(nz_true):.4f}")
