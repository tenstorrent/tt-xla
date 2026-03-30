# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
Mochi DiT inference with monkey-patches to eliminate torch.compile graph breaks.

The unpatched MochiTransformer3DModel produces ~25 separate graphs when compiled.
This script patches 5 methods across 4 diffusers classes to fix 7 graph break causes:
  1. Tensor.unflatten() → view()  (5 breaks)
  2. torch.nonzero() → static attention mask  (1+ breaks)
  3. torch.autocast context manager → explicit casting  (1 break)
  4. Conditional norm checks → unconditional calls  (2+ breaks)
  5. assert statements → removed  (1 break)
  6. @apply_lora_scale decorator → removed  (1 break)
  7. hasattr() check → context_pre_only attribute check  (1 break)

See mochi_dit_graph_break_report.md for full analysis.
"""

import torch
import torch.nn.functional as F
import torch_xla.core.xla_model as xm
import torch_xla.runtime as xr
from diffusers import MochiTransformer3DModel
from diffusers.models.attention_processor import MochiAttnProcessor2_0
from diffusers.models.embeddings import MochiAttentionPool
from diffusers.models.modeling_outputs import Transformer2DModelOutput
from diffusers.models.transformers.transformer_mochi import MochiRoPE


# ---------------------------------------------------------------------------
# Patch 1: MochiAttentionPool.pool_tokens — remove assert statements (Issue 5)
# Original: diffusers/models/embeddings.py:1974-1993
# ---------------------------------------------------------------------------
def patched_pool_tokens(x: torch.Tensor, mask: torch.Tensor, *, keepdim=False) -> torch.Tensor:
    # Removed assert statements that cause graph breaks:
    #   assert x.size(1) == mask.size(1)
    #   assert x.size(0) == mask.size(0)
    mask = mask[:, :, None].to(dtype=x.dtype)
    mask = mask / mask.sum(dim=1, keepdim=True).clamp(min=1)
    pooled = (x * mask).sum(dim=1, keepdim=keepdim)
    return pooled


# ---------------------------------------------------------------------------
# Patch 2: MochiAttentionPool.forward — unflatten → view (Issue 1a)
# Original: diffusers/models/embeddings.py:1995-2037
# ---------------------------------------------------------------------------
def patched_attention_pool_forward(self, x: torch.Tensor, mask: torch.BoolTensor) -> torch.Tensor:
    D = x.size(2)

    # Construct attention mask, shape: (B, 1, num_queries=1, num_keys=1+L).
    attn_mask = mask[:, None, None, :].bool()  # (B, 1, 1, L).
    attn_mask = F.pad(attn_mask, (1, 0), value=True)  # (B, 1, 1, 1+L).

    # Average non-padding token features. These will be used as the query.
    x_pool = self.pool_tokens(x, mask, keepdim=True)  # (B, 1, D)

    # Concat pooled features to input sequence.
    x = torch.cat([x_pool, x], dim=1)  # (B, L+1, D)

    # Compute queries, keys, values. Only the mean token is used to create a query.
    kv = self.to_kv(x)  # (B, L+1, 2 * D)
    q = self.to_q(x[:, 0])  # (B, D)

    # Extract heads — use view instead of unflatten to avoid graph breaks.
    head_dim = D // self.num_attention_heads
    kv = kv.view(kv.shape[0], kv.shape[1], 2, self.num_attention_heads, head_dim)  # (B, 1+L, 2, H, head_dim)
    kv = kv.transpose(1, 3)  # (B, H, 2, 1+L, head_dim)
    k, v = kv.unbind(2)  # (B, H, 1+L, head_dim)
    q = q.view(q.shape[0], self.num_attention_heads, head_dim)  # (B, H, head_dim)
    q = q.unsqueeze(2)  # (B, H, 1, head_dim)

    # Compute attention.
    x = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask, dropout_p=0.0)  # (B, H, 1, head_dim)

    # Concatenate heads and run output.
    x = x.squeeze(2).flatten(1, 2)  # (B, D = H * head_dim)
    x = self.to_out(x)
    return x


# ---------------------------------------------------------------------------
# Patch 3: MochiRoPE._create_rope — remove torch.autocast (Issue 3)
# Original: diffusers/models/transformers/transformer_mochi.py:285-292
# ---------------------------------------------------------------------------
def patched_create_rope(self, freqs: torch.Tensor, pos: torch.Tensor) -> torch.Tensor:
    # Removed torch.autocast context manager; explicit float32 casts are sufficient.
    freqs = torch.einsum("nd,dhf->nhf", pos.to(torch.float32), freqs.to(torch.float32))
    freqs_cos = torch.cos(freqs)
    freqs_sin = torch.sin(freqs)
    return freqs_cos, freqs_sin


# ---------------------------------------------------------------------------
# Patch 4: MochiAttnProcessor2_0.__call__ — the big patch
#   Issues 1b (unflatten→view), 2 (nonzero→static mask),
#          4 (conditional norms→unconditional), 7 (hasattr→context_pre_only)
# Original: diffusers/models/attention_processor.py:1005-1100
# ---------------------------------------------------------------------------
def patched_mochi_attn_processor_call(
    self,
    attn,
    hidden_states: torch.Tensor,
    encoder_hidden_states: torch.Tensor,
    attention_mask: torch.Tensor,
    image_rotary_emb=None,
) -> torch.Tensor:
    # Hidden-stream QKV projections.
    query = attn.to_q(hidden_states)
    key = attn.to_k(hidden_states)
    value = attn.to_v(hidden_states)

    # Issue 1b: use view instead of unflatten.
    head_dim = query.shape[2] // attn.heads
    query = query.view(query.shape[0], query.shape[1], attn.heads, head_dim)
    key = key.view(key.shape[0], key.shape[1], attn.heads, head_dim)
    value = value.view(value.shape[0], value.shape[1], attn.heads, head_dim)

    # Issue 4: call norms unconditionally (always initialized in MochiAttention).
    query = attn.norm_q(query)
    key = attn.norm_k(key)

    # Context-stream QKV projections.
    encoder_query = attn.add_q_proj(encoder_hidden_states)
    encoder_key = attn.add_k_proj(encoder_hidden_states)
    encoder_value = attn.add_v_proj(encoder_hidden_states)

    # Issue 1b: use view instead of unflatten.
    encoder_query = encoder_query.view(encoder_query.shape[0], encoder_query.shape[1], attn.heads, head_dim)
    encoder_key = encoder_key.view(encoder_key.shape[0], encoder_key.shape[1], attn.heads, head_dim)
    encoder_value = encoder_value.view(encoder_value.shape[0], encoder_value.shape[1], attn.heads, head_dim)

    # Issue 4: call norms unconditionally.
    encoder_query = attn.norm_added_q(encoder_query)
    encoder_key = attn.norm_added_k(encoder_key)

    # RoPE application (kept as-is, image_rotary_emb is always provided for Mochi).
    if image_rotary_emb is not None:

        def apply_rotary_emb(x, freqs_cos, freqs_sin):
            x_even = x[..., 0::2].float()
            x_odd = x[..., 1::2].float()
            cos = (x_even * freqs_cos - x_odd * freqs_sin).to(x.dtype)
            sin = (x_even * freqs_sin + x_odd * freqs_cos).to(x.dtype)
            return torch.stack([cos, sin], dim=-1).flatten(-2)

        query = apply_rotary_emb(query, *image_rotary_emb)
        key = apply_rotary_emb(key, *image_rotary_emb)

    # Transpose to (batch, heads, seq, dim).
    query, key, value = query.transpose(1, 2), key.transpose(1, 2), value.transpose(1, 2)
    encoder_query, encoder_key, encoder_value = (
        encoder_query.transpose(1, 2),
        encoder_key.transpose(1, 2),
        encoder_value.transpose(1, 2),
    )

    sequence_length = query.size(2)
    encoder_sequence_length = encoder_query.size(2)

    # Issue 2: replace torch.nonzero loop with static attention masking.
    # Concatenate all tokens (hidden + encoder) into a single sequence.
    full_query = torch.cat([query, encoder_query], dim=2)
    full_key = torch.cat([key, encoder_key], dim=2)
    full_value = torch.cat([value, encoder_value], dim=2)

    # Build additive attention mask: (batch, 1, 1, total_seq_len).
    # Hidden key positions are always valid; encoder key positions follow attention_mask.
    batch_size = query.shape[0]
    hidden_mask = torch.ones(
        batch_size, sequence_length, dtype=attention_mask.dtype, device=attention_mask.device
    )
    full_mask = torch.cat([hidden_mask, attention_mask], dim=1)
    # Convert 0/1 mask to additive mask: valid=0.0, padding=large_negative.
    attn_mask = full_mask[:, None, None, :].to(dtype=query.dtype)
    attn_mask = (1.0 - attn_mask) * torch.finfo(query.dtype).min

    attn_output = F.scaled_dot_product_attention(
        full_query, full_key, full_value, attn_mask=attn_mask, dropout_p=0.0, is_causal=False
    )

    # Split output back into hidden and encoder parts.
    hidden_states = attn_output[:, :, :sequence_length, :]
    encoder_hidden_states = attn_output[:, :, sequence_length:, :]

    # Reshape: (batch, heads, seq, dim) → (batch, seq, heads*dim).
    hidden_states = hidden_states.transpose(1, 2).flatten(2, 3)
    encoder_hidden_states = encoder_hidden_states.transpose(1, 2).flatten(2, 3)

    # Linear proj + dropout.
    hidden_states = attn.to_out[0](hidden_states)
    hidden_states = attn.to_out[1](hidden_states)

    # Issue 7: use context_pre_only instead of hasattr.
    if not attn.context_pre_only:
        encoder_hidden_states = attn.to_add_out(encoder_hidden_states)

    return hidden_states, encoder_hidden_states


# ---------------------------------------------------------------------------
# Patch 5: MochiTransformer3DModel.forward
#   Issues 1c (unflatten→view), 6 (remove @apply_lora_scale decorator)
# Original: diffusers/models/transformers/transformer_mochi.py:407-470
# ---------------------------------------------------------------------------
def patched_transformer_forward(
    self,
    hidden_states: torch.Tensor,
    encoder_hidden_states: torch.Tensor,
    timestep: torch.LongTensor,
    encoder_attention_mask: torch.Tensor,
    attention_kwargs=None,
    return_dict: bool = True,
) -> torch.Tensor:
    # Issue 6: this function is NOT decorated with @apply_lora_scale.
    # We don't use LoRA, so scale_lora_layers/unscale_lora_layers are unnecessary.

    batch_size, num_channels, num_frames, height, width = hidden_states.shape
    p = self.config.patch_size

    post_patch_height = height // p
    post_patch_width = width // p

    temb, encoder_hidden_states = self.time_embed(
        timestep,
        encoder_hidden_states,
        encoder_attention_mask,
        hidden_dtype=hidden_states.dtype,
    )

    hidden_states = hidden_states.permute(0, 2, 1, 3, 4).flatten(0, 1)
    hidden_states = self.patch_embed(hidden_states)
    # Issue 1c: use view instead of unflatten.
    num_patch_frames = hidden_states.shape[0] // batch_size
    hidden_states = hidden_states.view(
        batch_size, num_patch_frames, *hidden_states.shape[1:]
    ).flatten(1, 2)

    image_rotary_emb = self.rope(
        self.pos_frequencies,
        num_frames,
        post_patch_height,
        post_patch_width,
        device=hidden_states.device,
        dtype=torch.float32,
    )

    for i, block in enumerate(self.transformer_blocks):
        if torch.is_grad_enabled() and self.gradient_checkpointing:
            hidden_states, encoder_hidden_states = self._gradient_checkpointing_func(
                block,
                hidden_states,
                encoder_hidden_states,
                temb,
                encoder_attention_mask,
                image_rotary_emb,
            )
        else:
            hidden_states, encoder_hidden_states = block(
                hidden_states=hidden_states,
                encoder_hidden_states=encoder_hidden_states,
                temb=temb,
                encoder_attention_mask=encoder_attention_mask,
                image_rotary_emb=image_rotary_emb,
            )
    hidden_states = self.norm_out(hidden_states, temb)
    hidden_states = self.proj_out(hidden_states)

    hidden_states = hidden_states.reshape(
        batch_size, num_frames, post_patch_height, post_patch_width, p, p, -1
    )
    hidden_states = hidden_states.permute(0, 6, 1, 2, 4, 3, 5)
    output = hidden_states.reshape(batch_size, -1, num_frames, height, width)

    if not return_dict:
        return (output,)
    return Transformer2DModelOutput(sample=output)


# ---------------------------------------------------------------------------
# Apply all patches
# ---------------------------------------------------------------------------
def apply_patches():
    """Monkey-patch diffusers classes to eliminate graph breaks in Mochi DiT."""
    MochiAttentionPool.pool_tokens = staticmethod(patched_pool_tokens)
    MochiAttentionPool.forward = patched_attention_pool_forward
    MochiRoPE._create_rope = patched_create_rope
    MochiAttnProcessor2_0.__call__ = patched_mochi_attn_processor_call
    MochiTransformer3DModel.forward = patched_transformer_forward


# ---------------------------------------------------------------------------
# Model wrapper and inference (same as mochi_dit_sharded.py)
# ---------------------------------------------------------------------------
class MochiDiTWrapper(torch.nn.Module):
    """
    Wrapper for MochiTransformer3DModel that returns just the sample tensor.

    MochiTransformer3DModel returns Transformer2DModelOutput with .sample property,
    but we need a plain tensor output for compilation.
    """

    def __init__(self):
        super().__init__()
        self.transformer = MochiTransformer3DModel(num_layers=2)

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        timestep: torch.Tensor,
        encoder_attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        output = self.transformer(
            hidden_states=hidden_states,
            encoder_hidden_states=encoder_hidden_states,
            timestep=timestep,
            encoder_attention_mask=encoder_attention_mask,
        )
        return output.sample


def run_mochi_dit():
    """Run Mochi DiT on TT device."""
    # Instantiate model with random weights and reduced layers.
    model = MochiDiTWrapper().to(torch.bfloat16)

    # Put it in inference mode and compile it.
    model = model.eval()
    model.compile(backend="tt")

    # Input tensors matching full model activation shapes.
    hidden_states = torch.randn(1, 12, 2, 60, 106, dtype=torch.bfloat16)
    timestep = torch.tensor([500], dtype=torch.long)
    encoder_hidden_states = torch.randn(1, 128, 4096, dtype=torch.bfloat16)
    encoder_attention_mask = torch.ones(1, 128, dtype=torch.long)

    # Connect the device.
    device = xm.xla_device()

    # Move model and inputs to device.
    model = model.to(device)
    hidden_states = hidden_states.to(device)
    timestep = timestep.to(device)
    encoder_hidden_states = encoder_hidden_states.to(device)
    encoder_attention_mask = encoder_attention_mask.to(device)

    # Run inference.
    with torch.no_grad():
        output = model(
            hidden_states, encoder_hidden_states, timestep, encoder_attention_mask
        )

    return output


if __name__ == "__main__":
    # By default torch_xla uses the CPU device so we have to set it to TT device.
    xr.set_device_type("TT")

    apply_patches()
    output = run_mochi_dit()
    print(f"Output shape: {output.shape}")
    print(output)
