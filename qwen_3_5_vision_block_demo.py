# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
import torch
import torch_xla
import torch_xla.runtime as xr
from transformers import Qwen3_5VisionModel

xr.set_device_type("TT")
device = torch_xla.device()

model_id = "Qwen/Qwen3.5-27B"

# Load the full vision encoder to get pretrained weights, then extract a single block
vision_encoder = Qwen3_5VisionModel.from_pretrained(
    model_id, torch_dtype=torch.bfloat16, attn_implementation="eager"
)

config = vision_encoder.config
hidden_size = config.hidden_size  # 1152
num_heads = config.num_heads  # 16
head_dim = hidden_size // num_heads  # 72

vision_block = vision_encoder.blocks[0]
del vision_encoder

# Build inputs on CPU first for the golden reference run
seq_len = 16

hidden_states_cpu = torch.randn(seq_len, hidden_size, dtype=torch.bfloat16)
cu_seqlens_cpu = torch.tensor([0, seq_len], dtype=torch.int32)

pos_emb_cpu = torch.randn(seq_len, head_dim, dtype=torch.bfloat16)
position_embeddings_cpu = (pos_emb_cpu.cos(), pos_emb_cpu.sin())

# Golden reference: run the block on CPU
vision_block.eval()
with torch.no_grad():
    expected = vision_block(
        hidden_states_cpu,
        cu_seqlens=cu_seqlens_cpu,
        position_embeddings=position_embeddings_cpu,
    )

# Move block to TT device, compile, and run with the same inputs
vision_block = vision_block.to(device)
compiled_block = torch.compile(vision_block, backend="tt")

hidden_states = hidden_states_cpu.to(device)
cu_seqlens = cu_seqlens_cpu.to(device)
position_embeddings = (
    position_embeddings_cpu[0].to(device),
    position_embeddings_cpu[1].to(device),
)

with torch.no_grad():
    actual = compiled_block(
        hidden_states,
        cu_seqlens=cu_seqlens,
        position_embeddings=position_embeddings,
    )

# PCC (Pearson Correlation Coefficient) test
actual_cpu = actual.cpu().float()
expected_cpu = expected.float()

pcc = torch.corrcoef(torch.stack([actual_cpu.flatten(), expected_cpu.flatten()]))[
    0, 1
].item()

PCC_THRESHOLD = 0.99
passed = pcc >= PCC_THRESHOLD

print(f"Model: {model_id} (single vision block)")
print(f"Input shape:  ({seq_len}, {hidden_size})")
print(f"Output shape: {tuple(actual.shape)}")
print(f"PCC: {pcc:.6f}  (threshold: {PCC_THRESHOLD})")
print(f"Result: {'PASS' if passed else 'FAIL'}")

assert passed, f"PCC {pcc:.6f} is below threshold {PCC_THRESHOLD}"
