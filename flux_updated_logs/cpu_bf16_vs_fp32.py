# SPDX-License-Identifier: Apache-2.0
"""
PURE-CPU decomposition of the FLUX.2 transformer PCC drop.

We already have TT-bf16 vs CPU-bf16 (the sweep): 0.980 @NS=24 ... 0.650 @NS=48.
The open fork: is that drop the model's INHERENT bf16 instability at depth, or error
the TT device adds beyond bf16? This script measures the model-inherent piece with NO
device involved: CPU-bf16 forward vs CPU-fp32 forward, at increasing single-block depth.

  CPU-bf16 vs CPU-fp32  ~0.99  -> model is bf16-stable; the 0.65 is TT device error to chase.
  CPU-bf16 vs CPU-fp32  ~0.65  -> model is fundamentally bf16-unstable at depth; fix = higher
                                  precision in the deep single blocks; more chips won't help.

Full dual stack (NL=8); sweeps NS over the deep single blocks. Inputs built once in fp32 and
cast to bf16 so both forwards see identical content.
"""
import torch

from third_party.tt_forge_models.flux2.pytorch import ModelLoader, ModelVariant

NL = 8
NS_LIST = [12, 24, 46, 48]
torch.manual_seed(0)


def pcc(a, b):
    a = a.detach().to(torch.float32).flatten()
    b = b.detach().to(torch.float32).flatten()
    a = a - a.mean()
    b = b - b.mean()
    denom = (a.norm() * b.norm()).item()
    return float("nan") if denom == 0 else float((a @ b).item() / denom)


def truncate(model, nl, ns):
    t = model.transformer
    t.transformer_blocks = t.transformer_blocks[:nl]
    t.single_transformer_blocks = t.single_transformer_blocks[:ns]
    return model


print(">>> loading fp32 transformer ...", flush=True)
m_fp32 = ModelLoader(ModelVariant.TRANSFORMER).load_model(dtype_override=torch.float32)
print(">>> loading bf16 transformer ...", flush=True)
m_bf16 = ModelLoader(ModelVariant.TRANSFORMER).load_model(dtype_override=torch.bfloat16)

print(">>> building inputs (fp32) ...", flush=True)
loader = ModelLoader(ModelVariant.TRANSFORMER)
inp_fp32 = loader.load_inputs(dtype_override=torch.float32)
inp_bf16 = [x.to(torch.bfloat16) for x in inp_fp32]

print(f"\n{'NS':>4} | {'CPU-bf16 vs CPU-fp32':>22} | note", flush=True)
print("-" * 60, flush=True)
results = []
# descending so we slice both models down progressively (in-place truncation)
for ns in sorted(NS_LIST, reverse=True):
    truncate(m_fp32, NL, ns)
    truncate(m_bf16, NL, ns)
    with torch.no_grad():
        out_fp32 = m_fp32(*inp_fp32)
        out_bf16 = m_bf16(*inp_bf16)
    p = pcc(out_bf16, out_fp32)
    results.append((ns, p))
    print(f"{ns:>4} | {p:>22.6f} |", flush=True)

print("\n=== SUMMARY (CPU-bf16 vs CPU-fp32, NL=8) ===", flush=True)
for ns, p in sorted(results):
    print(f"  NS={ns:>3}  inherent_bf16_PCC = {p:.6f}", flush=True)
print("CPU_DECOMP_DONE", flush=True)
