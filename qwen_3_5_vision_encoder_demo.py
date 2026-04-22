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

vision_encoder = Qwen3_5VisionModel.from_pretrained(
    model_id, torch_dtype=torch.bfloat16, attn_implementation="eager"
)

config = vision_encoder.config
patch_dim = (
    config.in_channels
    * config.temporal_patch_size
    * config.patch_size
    * config.patch_size
)

# Simulate a single image: temporal=1, 2x2 patch grid → 4 total patches
# (reduced from 4x4 to isolate whether channel-blocking or spatial-blocking
# dominates the Conv3d L1 overflow — see qwen_3_5_issue.md)
# grid_thw_cpu = torch.tensor([[1, 2, 2]], dtype=torch.long)
grid_thw_cpu = torch.tensor([[1, 4, 4]], dtype=torch.long)
total_patches = grid_thw_cpu.prod(dim=-1).sum().item()

hidden_states_cpu = torch.randn(total_patches, patch_dim, dtype=torch.bfloat16)

# Golden reference: run the full encoder on CPU
vision_encoder.eval()
with torch.no_grad():
    expected = vision_encoder(hidden_states_cpu, grid_thw=grid_thw_cpu)

# Move encoder to TT device, compile, and run with the same inputs
vision_encoder = vision_encoder.to(device)
compiled_encoder = torch.compile(vision_encoder, backend="tt")
print("Compiled model")

hidden_states = hidden_states_cpu.to(device)
grid_thw = grid_thw_cpu.to(device)

with torch.no_grad():
    actual = compiled_encoder(hidden_states, grid_thw=grid_thw)

# PCC (Pearson Correlation Coefficient) test on the merged pooler output
actual_pooler = actual.pooler_output.cpu().float()
expected_pooler = expected.pooler_output.float()

pcc = torch.corrcoef(torch.stack([actual_pooler.flatten(), expected_pooler.flatten()]))[
    0, 1
].item()

merged_patches = total_patches // (config.spatial_merge_size**2)

PCC_THRESHOLD = 0.99
passed = pcc >= PCC_THRESHOLD

print(f"Model: {model_id} (full vision encoder)")
print(f"Input pixel patches shape: ({total_patches}, {patch_dim})")
print(f"Grid (T, H, W): {grid_thw_cpu.tolist()}")
print(f"Output pooler shape: {tuple(actual_pooler.shape)}")
print(f"Expected: {merged_patches} merged tokens x {config.out_hidden_size} dims")
print(f"PCC: {pcc:.6f}  (threshold: {PCC_THRESHOLD})")
print(f"Result: {'PASS' if passed else 'FAIL'}")

assert passed, f"PCC {pcc:.6f} is below threshold {PCC_THRESHOLD}"
