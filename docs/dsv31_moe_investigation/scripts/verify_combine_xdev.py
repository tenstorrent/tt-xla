"""Correct cross-device all_to_all_combine golden vs device dump (FABRIC_1D).

Mesh (4 rows x 8 cols), experts 1D-mapped: expert e -> device e//8, local e%8.
Combine gathers along the cluster axis (rows) within each column. So device d
(column c=d%8) collects expert outputs for all experts whose owning device is in
column c. combined[d][k, t] = down_all[owner(e), e%8, t] if owner(e)%8 == d%8,
where e = metadata[t,k]; else 0. (token slot for token t is t, first TPD slots.)
"""
import torch
D = "tmp/op_dump_fab1d"
L = lambda p: torch.load(f"{D}/{p}.pt", map_location="cpu")
def pcc(a, b):
    a=a.float().flatten()-a.float().mean(); b=b.float().flatten()-b.float().mean()
    d=a.norm()*b.norm(); return (torch.dot(a,b)/d).item() if d>0 else float("nan")

meta = L("comb0_in1_all").long()[:, 0, 0]   # [32, 4096, 8] (all-gathered, identical)
down = L("comb0_in0_all")                    # [32, 8, 1, 4096, 7168]
devout = L("comb0_out0_all")                 # [32, 8, 1, 1024, 7168]
ND, ELOC = down.shape[0], down.shape[1]
TPD = devout.shape[3]; K = devout.shape[1]; H = down.shape[-1]
COLS = 8  # mesh columns
tidx = torch.arange(TPD)
md = meta[0, :TPD]                            # [TPD, K] global expert ids (same all dev)
e_owner = md // ELOC                          # [TPD,K] owning device
e_local = md % ELOC
e_col = e_owner % COLS

pccs = []
for d in range(ND):
    dcol = d % COLS
    dv = devout[d, :, 0, :, :].float()        # [K, TPD, H]
    gold = torch.zeros(K, TPD, H)
    for k in range(K):
        owner = e_owner[:, k]; loc = e_local[:, k]; valid = (e_col[:, k] == dcol)
        gathered = down[owner, loc, 0, tidx, :].float()   # [TPD, H] cross-device gather
        gold[k] = gathered * valid.unsqueeze(-1)
    pccs.append(pcc(dv, gold))
    if d < 4:
        print(f"dev{d}(col{dcol}): PCC={pccs[-1]:.5f} nz_dev={(dv.abs().sum(-1)>0).float().mean():.4f} "
              f"nz_gold={(gold.abs().sum(-1)>0).float().mean():.4f}")
import statistics as st
print(f"COMBINE xdev golden: mean PCC = {st.mean(pccs):.5f} (min {min(pccs):.4f} max {max(pccs):.4f})")
