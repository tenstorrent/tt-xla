# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
import torch
import torch_xla
import torch_xla.runtime as xr
from transformers import AutoModelForCausalLM, AutoTokenizer

xr.set_device_type("TT")
device = torch_xla.device()

model_id = "inclusionAI/LLaDA2.0-mini"

tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    model_id, trust_remote_code=True, torch_dtype="auto"
)

model = model.to(device)
model.eval()

compiled_model = torch.compile(model, backend="tt")

prompt = "The capital of France is"
inputs = tokenizer(prompt, return_tensors="pt").to(device)

# LLaDA2.0 is a diffusion LM and requires a 4-D block attention mask of shape
# (batch_size, 1, seq_length, seq_length) -- it does not accept None or the
# usual 2-D HF mask. See modeling_llada2_moe.py:866-876.
input_ids = inputs["input_ids"]
batch_size, seq_length = input_ids.shape
attention_mask = torch.ones(
    batch_size, 1, seq_length, seq_length, dtype=torch.bool, device=device
)

with torch.no_grad():
    outputs = compiled_model(input_ids, attention_mask=attention_mask)
    logits = outputs.logits[:, -1, :]
    probs = torch.softmax(logits, dim=-1)
    top5_probs, top5_indices = torch.topk(probs, 5, dim=-1)

print(f"Prompt: `{prompt}`")
print(f"Top prediction: `{tokenizer.decode(top5_indices[0][0])}`")

print(f"\n{'Rank':<6} {'Token':<15} {'Probability':<12}")
print("-" * 35)
for rank, (idx, prob) in enumerate(zip(top5_indices[0], top5_probs[0]), start=1):
    token = tokenizer.decode(idx)
    print(f"{rank:<6} {repr(token):<15} {prob.item():.4%}")