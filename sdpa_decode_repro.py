"""Minimal single-device ttnn repro to test scaled_dot_product_attention_decode
at the exact per-shard shapes seen in the llama_3_1_70b_tp_qb2 2D-mesh decode
graph (b=16, nh=32 q-heads, nkv=4 kv-heads, s=128, dh=128).

SDPA-decode is a LOCAL op (no cross-device comm), so the per-device shard on the
(2,2) mesh is numerically identical to running these shapes single-device. If the
op is correct here it is correct on the mesh; a low PCC here would be a genuine
metal-op bug at these shapes.
"""
import math
import torch
import ttnn

torch.manual_seed(0)

B, NH, NKV, S, DH = 16, 32, 4, 128, 128   # 2D per-shard shape
CUR = 100                                  # valid cached positions [0, CUR]
SCALE = 1.0 / math.sqrt(DH)                # 0.0883883...
GROUP = NH // NKV                          # GQA repeat factor

# ---- inputs (bf16 like the model) ----
q  = torch.randn(1, B, NH, DH, dtype=torch.bfloat16)
k  = torch.randn(B, NKV, S, DH, dtype=torch.bfloat16)
v  = torch.randn(B, NKV, S, DH, dtype=torch.bfloat16)
# additive attn mask [b,1,nh,s]: 0 for valid key cols (<=CUR), -inf beyond
mask = torch.zeros(B, 1, NH, S, dtype=torch.bfloat16)
mask[:, :, :, CUR + 1:] = float("-inf")

# ---- torch reference (fp32 math) ----
def torch_ref():
    qf = q.float()[0]                       # [B,NH,DH]
    kf = k.float()                          # [B,NKV,S,DH]
    vf = v.float()
    # expand kv heads to q heads (GQA)
    kf = kf.repeat_interleave(GROUP, dim=1) # [B,NH,S,DH]
    vf = vf.repeat_interleave(GROUP, dim=1)
    scores = torch.einsum("bhd,bhsd->bhs", qf, kf) * SCALE   # [B,NH,S]
    scores = scores + mask.float()[:, 0, :, :]               # [B,NH,S]
    attn = torch.softmax(scores, dim=-1)
    out = torch.einsum("bhs,bhsd->bhd", attn, vf)            # [B,NH,DH]
    return out                                               # [B,NH,DH]

ref = torch_ref()

# ---- ttnn device run ----
dev = ttnn.open_device(device_id=0)
try:
    tq = ttnn.from_torch(q, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev)
    tk = ttnn.from_torch(k, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev)
    tv = ttnn.from_torch(v, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev)
    tm = ttnn.from_torch(mask, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev)
    out = ttnn.transformer.scaled_dot_product_attention_decode(
        tq, tk, tv, is_causal=False, attn_mask=tm, cur_pos=[CUR] * B, scale=SCALE,
    )
    dev_out = ttnn.to_torch(out).float()   # [1,B,NH,DH]
finally:
    ttnn.close_device(dev)

dev_out = dev_out.reshape(B, NH, DH)

# ---- PCC ----
def pcc(a, b):
    a = a.flatten().double(); b = b.flatten().double()
    a = torch.nan_to_num(a); b = torch.nan_to_num(b)
    va = a - a.mean(); vb = b - b.mean()
    denom = (va.norm() * vb.norm()).item()
    if denom == 0:
        return float("nan")
    return (torch.dot(va, vb).item() / denom)

print(f"shapes: Q{list(q.shape)} K{list(k.shape)} V{list(v.shape)} mask{list(mask.shape)} cur_pos={CUR}")
print(f"ref  mean={ref.mean().item():.5f} std={ref.std().item():.5f}")
print(f"dev  mean={dev_out.mean().item():.5f} std={dev_out.std().item():.5f}  nan={torch.isnan(dev_out).any().item()} inf={torch.isinf(dev_out).any().item()}")
print(f"PCC(ttnn sdpa_decode vs torch) = {pcc(ref, dev_out):.6f}")
