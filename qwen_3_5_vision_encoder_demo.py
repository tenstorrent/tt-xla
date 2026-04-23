# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
import torch
import torch_xla
import torch_xla.runtime as xr
from transformers import Qwen3_5VisionModel
from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5VisionConfig


def patch_qwen_vision_single_sequence_attention() -> None:
    """Avoid varlen Python list conversion for single-sequence demos."""
    from transformers.models.qwen3_5 import modeling_qwen3_5 as qwen_modeling

    if getattr(
        qwen_modeling.Qwen3_5VisionAttention.forward,
        "_ttxla_single_sequence_patch",
        False,
    ):
        return

    original_forward = qwen_modeling.Qwen3_5VisionAttention.forward

    def patched_forward(
        self,
        hidden_states: torch.Tensor,
        cu_seqlens: torch.Tensor,
        rotary_pos_emb: torch.Tensor | None = None,
        position_embeddings: tuple[torch.Tensor, torch.Tensor] | None = None,
        **kwargs,
    ) -> torch.Tensor:
        if (
            not qwen_modeling.is_flash_attention_requested(self.config)
            and cu_seqlens.numel() == 2
        ):
            seq_length = hidden_states.shape[0]
            query_states, key_states, value_states = (
                self.qkv(hidden_states)
                .reshape(seq_length, 3, self.num_heads, -1)
                .permute(1, 0, 2, 3)
                .unbind(0)
            )
            cos, sin = position_embeddings
            query_states, key_states = qwen_modeling.apply_rotary_pos_emb_vision(
                query_states, key_states, cos, sin
            )

            query_states = query_states.transpose(0, 1).unsqueeze(0)
            key_states = key_states.transpose(0, 1).unsqueeze(0)
            value_states = value_states.transpose(0, 1).unsqueeze(0)

            attention_interface = qwen_modeling.ALL_ATTENTION_FUNCTIONS.get_interface(
                self.config._attn_implementation,
                qwen_modeling.eager_attention_forward,
            )
            attn_output, _ = attention_interface(
                self,
                query_states,
                key_states,
                value_states,
                attention_mask=None,
                scaling=self.scaling,
                dropout=0.0 if not self.training else self.attention_dropout,
                is_causal=False,
                **kwargs,
            )

            attn_output = attn_output.reshape(seq_length, -1).contiguous()
            return self.proj(attn_output)

        return original_forward(
            self,
            hidden_states,
            cu_seqlens,
            rotary_pos_emb=rotary_pos_emb,
            position_embeddings=position_embeddings,
            **kwargs,
        )

    patched_forward._ttxla_single_sequence_patch = True
    qwen_modeling.Qwen3_5VisionAttention.forward = patched_forward


xr.set_device_type("TT")
device = torch_xla.device()

model_id = "Qwen/Qwen3.5-27B"

patch_qwen_vision_single_sequence_attention()

# -- real model --
vision_encoder = Qwen3_5VisionModel.from_pretrained(
    model_id, torch_dtype=torch.bfloat16, attn_implementation="eager"
)
config = vision_encoder.config

# -- fake model with variable hidden size and random weights --
# Load config and reduce hidden_size to shrink Conv3d output channels
# config = Qwen3_5VisionConfig.from_pretrained(model_id)
# config.hidden_size = 1152  # Reduced from 1152 (halves Conv3d size)

# # Initialize with modified config (random weights, not pretrained)
# vision_encoder = Qwen3_5VisionModel(config).to(torch.bfloat16)

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
