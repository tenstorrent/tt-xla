"""ROOT-CAUSE PROOF for deepseek-v3.1 low PCC (issue #5096).

all_to_all_dispatch drops the SECOND HALF of the batch dimension. Half the
tokens receive zero routed-MoE output -> ~0.5 routed PCC -> garbage generation.

Run the diverse op-dump first:
  OP_DUMP_DIR=tmp/op_dump_diverse DUMP_DIVERSE=1 \
    pytest -q tests/torch/models/deepseek_v3_1/test_deepseek_v3_1.py::test_dump_sparse_matmul_io[64]

Then this script proves the drop from the dumps (no device needed).
"""
import torch

D = "tmp/op_dump_diverse"
L = lambda p: torch.load(f"{D}/{p}.pt", map_location="cpu")

# 1) dispatch INPUT has all 64 batches with valid routing
h = L("disp0_in0")[:, 0, :, :].float()
idx = L("disp0_in1").long()[:, 0, :, :]
print(f"dispatch INPUT hidden mag: batches[0:32]={h[:32].abs().sum():.0f} "
      f"batches[32:64]={h[32:].abs().sum():.0f}  (both nonzero => all present)")

# 2) dispatch OUTPUT metadata is exactly 50% populated
dm = L("disp0_out1_all").long()           # [32,1,BD,S,K]
dm0 = dm[0].reshape(-1, dm.shape[-1])
print(f"dispatch metadata nonzero rows: {int((dm0!=0).any(-1).sum())}/{dm0.shape[0]} "
      f"= {(dm0!=0).any(-1).float().mean():.0%}")

# 3) which original tokens survive (present anywhere across all 32 device metadata)
cm = L("comb0_in1_all").long()[:, 0, 0].reshape(-1, 8)
uniq = set(map(tuple, torch.unique(cm, dim=0).tolist()))
present = torch.tensor([[tuple(idx[b, s].tolist()) in uniq for s in range(16)]
                        for b in range(64)])
pb = present.sum(1)
print(f"surviving batches: 0-31={int((pb[:32]>0).sum())}/32  "
      f"32-63={int((pb[32:]>0).sum())}/32")

# 4) combine OUTPUT magnitude per batch-half (the actual routed output)
co = L("comb0_out0_all")                   # [32,K,1,1024,H]
mag = co.abs().sum(dim=(0, 1, 2, 4)).reshape(64, 16).sum(1)
print(f"combine OUTPUT mag: batches[0:32]mean={mag[:32].mean():.0f}  "
      f"batches[32:64]mean={mag[32:].mean():.1f}  (2nd half ~0 => DROPPED)")
