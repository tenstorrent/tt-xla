"""Run the A2aSparseMLP DEVICE path on CPU (custom ops' CPU impls) and compare
its routed-only output to the original DeepseekV3MoE golden. Discriminates
algorithm bug (low PCC on CPU) vs hardware-kernel numerics (high PCC on CPU,
divergence only on device). Dumps per-op golden intermediates to SPARSE_DUMP_DIR.
"""
import os
import torch

DUMP = "/localdev/sshon/tt-xla/tmp/sparse_dump_cpu"
if os.environ.get("ENABLE_DUMP") == "1":
    os.makedirs(DUMP, exist_ok=True)
    os.environ["SPARSE_DUMP_DIR"] = DUMP

from third_party.tt_forge_models.deepseek.deepseek_v3_1.pytorch.loader import ModelLoader

torch.manual_seed(0)
BATCH = int(os.environ.get("GOLDEN_BATCH", "64"))
# SEQ: "nopad" -> real benchmark prefill (~16 tokens, no pad, flat-fallback path);
# or an int -> padded to that length (e.g. 32 = split_seq path, regression check).
SEQ = os.environ.get("GOLDEN_SEQ", "nopad")


def comp_pcc(a, b):
    a = a.detach().to(torch.float32).flatten()
    b = b.detach().to(torch.float32).flatten()
    a = a - a.mean()
    b = b - b.mean()
    denom = a.norm() * b.norm()
    return (torch.dot(a, b) / denom).item() if denom > 0 else float("nan")


loader = ModelLoader(num_layers=4)
model = loader.load_model(dtype_override=torch.bfloat16)
config = loader.config
model.eval()
moe_block = model.model.layers[config.first_k_dense_replace].mlp
moe_block.eval()

# --- capture REAL layer-3 input ---
if SEQ == "nopad":
    if loader.tokenizer is None:
        loader._load_tokenizer()
    prompt = "Here is an exaustive list of the best practices for writing clean code:"
    toks = loader.tokenizer(prompt, return_tensors="pt").input_ids[0]
    input_ids = toks.unsqueeze(0).expand(BATCH, -1).contiguous()
else:
    input_ids = loader.load_inputs(batch_size=BATCH, seq_len=int(SEQ))
print(f"SEQ={SEQ} input_ids shape: {tuple(input_ids.shape)}")
cap = {}
h = moe_block.register_forward_pre_hook(lambda m, a: cap.__setitem__("x", a[0].detach().clone()))
with torch.no_grad():
    model.model(input_ids=input_ids, use_cache=False)
h.remove()
hidden = cap["x"]
print(f"hidden: {tuple(hidden.shape)} {hidden.dtype} absmax={hidden.abs().max():.2f}")

# --- golden routed-only = original_moe(x) - shared(x) ---
with torch.no_grad():
    full, _ = moe_block.mlp._cpu_forward(hidden)  # original MoE: routed + shared
    shared = moe_block.shared_experts(hidden)
    golden_routed = full - shared

# --- device-path on CPU (forced) -> routed-only ---
os.environ["FORCE_DEVICE_PATH"] = "1"
with torch.no_grad():
    dev_cpu_routed, _ = moe_block.mlp(hidden)
os.environ.pop("FORCE_DEVICE_PATH")

pcc = comp_pcc(golden_routed, dev_cpu_routed)
print("=" * 60)
print(f"device-path-on-CPU routed  vs  original-MoE routed golden")
print(f"  PCC = {pcc:.6f}")
print(f"  golden absmax={golden_routed.abs().max():.3f}  devCPU absmax={dev_cpu_routed.abs().max():.3f}")
print(f"  dumped intermediates -> {DUMP}/  (01_dispatched .. 05_output)")
print("=" * 60)
