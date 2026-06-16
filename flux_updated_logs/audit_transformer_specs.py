"""Audit shard_transformer_specs coverage on a meta-device Flux2Transformer2DModel.

No weights are downloaded/materialized — we build from config on the meta device, so
param.numel()/shape are known but cost ~0 memory. Then we classify every parameter as
fully-sharded, partially-replicated, or fully-replicated and rank replicated weights by size.
"""
import torch
from diffusers import Flux2Transformer2DModel

import sys
sys.path.insert(0, "third_party")
from tt_forge_models.flux2.pytorch.src.model_utils import (
    REPO_ID,
    shard_transformer_specs,
)

config = Flux2Transformer2DModel.load_config(REPO_ID, subfolder="transformer")
with torch.device("meta"):
    model = Flux2Transformer2DModel.from_config(config)

specs = shard_transformer_specs(model)
# Map param tensor id -> spec
spec_by_id = {id(p): s for p, s in specs.items()}

BYTES = 2  # bf16
NDEV = 8


def classify(spec):
    if spec is None:
        return "REPLICATED(not-in-spec)"
    if all(a is None for a in spec):
        return "REPLICATED(None...)"
    return "SHARDED"


rows = []
for name, p in model.named_parameters():
    spec = spec_by_id.get(id(p), None)
    kind = classify(spec)
    numel = p.numel()
    rows.append((name, numel, kind, spec))

total = sum(r[1] for r in rows)
sharded = sum(r[1] for r in rows if r[2] == "SHARDED")
repl = sum(r[1] for r in rows if r[2].startswith("REPLICATED"))

print(f"#params tensors: {len(rows)}")
print(f"total params: {total/1e9:.2f} B  ({total*BYTES/1e9:.1f} GB bf16)")
print(f"  sharded:    {sharded/1e9:.2f} B  -> {sharded*BYTES/NDEV/1e9:.2f} GB/device at {NDEV}-way")
print(f"  replicated: {repl/1e9:.3f} B  -> {repl*BYTES/1e9:.3f} GB/device (full copy on EVERY device)")
print()
print("=== top replicated weights by size (these waste device DRAM) ===")
repl_rows = sorted([r for r in rows if r[2].startswith("REPLICATED")], key=lambda r: -r[1])
for name, numel, kind, spec in repl_rows[:25]:
    if numel < 1_000_000:
        break
    print(f"  {numel*BYTES/1e6:8.1f} MB  {kind:24s} {name}  shape={tuple(model.get_parameter(name).shape)}")

# also estimate per-device DRAM from weights only
est = sharded * BYTES / NDEV + repl * BYTES
print()
print(f"Estimated weight DRAM/device at {NDEV}-way: {est/1e9:.2f} GB "
      f"(sharded {sharded*BYTES/NDEV/1e9:.2f} + replicated {repl*BYTES/1e9:.3f})")
