# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Standalone PyTorch reference model for the Z-Image text encoder (Qwen3).

Self-contained -- no imports from sibling directories.
Contains TextEncoderModule (wrapping Qwen3 decoder layers) plus helpers
for loading from HuggingFace and preparing inputs.

The text encoder in the Z-Image pipeline is a Qwen3 model accessed via
pipe.text_encoder. TextEncoderModule wraps it to run all decoder layers
directly, avoiding the hook-based output_hidden_states mechanism that
breaks torch.compile.

Returns the output of the last decoder layer (before final RMSNorm),
equivalent to text_encoder(..., output_hidden_states=True).hidden_states[-2].
"""

import os
from pathlib import Path

import torch
import torch.nn as nn

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

MODEL_ID = "Tongyi-MAI/Z-Image"
DTYPE = torch.bfloat16
MODEL_CACHE_PATH = "z_image_text_encoder.pt"
SEQ_LEN = 512
DIR = Path(os.path.dirname(os.path.abspath(__file__)))


# ---------------------------------------------------------------------------
# TextEncoderModule
# ---------------------------------------------------------------------------

class TextEncoderModule(nn.Module):
    """Wraps Qwen3 to run all decoder layers directly, avoiding the
    hook-based output_hidden_states mechanism that breaks torch.compile.

    Returns the output of the last decoder layer (before final RMSNorm),
    which is equivalent to
    text_encoder(..., output_hidden_states=True).hidden_states[-2].

    Why [-2] == last layer output: HuggingFace collects hidden_states as
    [input_embeds, layer_0_out, ..., layer_{N-1}_out] then replaces the
    last entry with the normed last_hidden_state, so [-2] is layer N-1's
    raw output.
    """

    def __init__(self, text_encoder):
        super().__init__()
        self.embed_tokens = text_encoder.embed_tokens
        self.rotary_emb = text_encoder.rotary_emb
        num_layers = text_encoder.config.num_hidden_layers
        self.layers = text_encoder.layers[: num_layers - 1]
        self.config = text_encoder.config
        self.has_sliding_layers = text_encoder.has_sliding_layers

    def forward(self, input_ids, attention_mask):
        inputs_embeds = self.embed_tokens(input_ids)

        cache_position = torch.arange(
            inputs_embeds.shape[1], device=inputs_embeds.device
        )
        position_ids = cache_position.unsqueeze(0)

        # Build causal masks directly as tensors to avoid HuggingFace's
        # flex_attention mask functions which use vmap (incompatible with XLA).
        seq_len = inputs_embeds.shape[1]
        dtype = inputs_embeds.dtype
        causal = torch.tril(
            torch.ones(seq_len, seq_len, dtype=torch.bool, device=inputs_embeds.device)
        )
        # Combine with padding mask: [1, 1, seq_len, seq_len] & [B, 1, 1, seq_len]
        full_mask = causal.unsqueeze(0).unsqueeze(0) & attention_mask.bool()[:, None, None, :]
        full_mask = torch.where(
            full_mask, 0.0, torch.finfo(dtype).min
        ).to(dtype)

        causal_mask_mapping = {"full_attention": full_mask}

        if self.has_sliding_layers:
            window = self.config.sliding_window
            row_idx = torch.arange(seq_len, device=inputs_embeds.device).unsqueeze(1)
            col_idx = torch.arange(seq_len, device=inputs_embeds.device).unsqueeze(0)
            sliding = causal & ((row_idx - col_idx) < window)
            sliding_mask = sliding.unsqueeze(0).unsqueeze(0) & attention_mask.bool()[:, None, None, :]
            sliding_mask = torch.where(
                sliding_mask, 0.0, torch.finfo(dtype).min
            ).to(dtype)
            causal_mask_mapping["sliding_attention"] = sliding_mask

        hidden_states = inputs_embeds
        position_embeddings = self.rotary_emb(hidden_states, position_ids)

        for decoder_layer in self.layers:
            hidden_states = decoder_layer(
                hidden_states,
                attention_mask=causal_mask_mapping[decoder_layer.attention_type],
                position_embeddings=position_embeddings,
                position_ids=position_ids,
            )

        return hidden_states

    def state_dict_for_ttnn(self):
        """Return state_dict with clean PyTorch names for TTNN parameter loading.

        Returns a dict with keys like:
          "embed_tokens.weight", "rotary_emb.inv_freq",
          "layers.0.input_layernorm.weight",
          "layers.0.self_attn.q_proj.weight", etc.
        """
        sd = {}

        # Embedding table
        sd["embed_tokens.weight"] = self.embed_tokens.weight.data

        # RoPE inverse frequencies
        sd["rotary_emb.inv_freq"] = self.rotary_emb.inv_freq.data

        # Decoder layers
        for i, layer in enumerate(self.layers):
            prefix = f"layers.{i}"

            # LayerNorm weights
            sd[f"{prefix}.input_layernorm.weight"] = layer.input_layernorm.weight.data
            sd[f"{prefix}.post_attention_layernorm.weight"] = layer.post_attention_layernorm.weight.data

            # Attention weights
            attn = layer.self_attn
            sd[f"{prefix}.self_attn.q_proj.weight"] = attn.q_proj.weight.data
            sd[f"{prefix}.self_attn.k_proj.weight"] = attn.k_proj.weight.data
            sd[f"{prefix}.self_attn.v_proj.weight"] = attn.v_proj.weight.data
            sd[f"{prefix}.self_attn.o_proj.weight"] = attn.o_proj.weight.data
            sd[f"{prefix}.self_attn.q_norm.weight"] = attn.q_norm.weight.data
            sd[f"{prefix}.self_attn.k_norm.weight"] = attn.k_norm.weight.data

            # MLP weights
            mlp = layer.mlp
            sd[f"{prefix}.mlp.gate_proj.weight"] = mlp.gate_proj.weight.data
            sd[f"{prefix}.mlp.up_proj.weight"] = mlp.up_proj.weight.data
            sd[f"{prefix}.mlp.down_proj.weight"] = mlp.down_proj.weight.data

        return sd


# ---------------------------------------------------------------------------
# Loading helpers
# ---------------------------------------------------------------------------

def _load_text_encoder():
    """Load the text encoder from HuggingFace Z-Image pipeline.

    Caches the TextEncoderModule state to disk after first load.
    """
    cache_path = DIR / MODEL_CACHE_PATH

    if cache_path.exists():
        print(f"Loading cached text encoder from {cache_path}")
        checkpoint = torch.load(cache_path, map_location="cpu", weights_only=False)
        # Reconstruct by loading the pipeline again for the model structure
        # then loading saved weights
        from diffusers import DiffusionPipeline

        pipe = DiffusionPipeline.from_pretrained(
            MODEL_ID, torch_dtype=DTYPE, trust_remote_code=True
        )
        model = TextEncoderModule(pipe.text_encoder)
        model.load_state_dict(checkpoint)
        del pipe
        return model

    print(f"Loading Z-Image pipeline from {MODEL_ID}...")
    from diffusers import DiffusionPipeline

    pipe = DiffusionPipeline.from_pretrained(
        MODEL_ID, torch_dtype=DTYPE, trust_remote_code=True
    )
    model = TextEncoderModule(pipe.text_encoder)

    # Cache for next time
    print(f"Caching text encoder to {cache_path}")
    torch.save(model.state_dict(), cache_path)

    del pipe
    return model


def get_input():
    """Get sample input_ids and attention_mask for the text encoder.

    Returns:
        (input_ids, attention_mask) both shape [1, 512]
        input_ids: INT64 (torch default), attention_mask: BFLOAT16
    """
    input_cache = DIR / "z_image_text_encoder_inputs.pt"

    if input_cache.exists():
        print(f"Loading cached inputs from {input_cache}")
        data = torch.load(input_cache, map_location="cpu", weights_only=True)
        return data["input_ids"], data["attention_mask"]

    print("Generating sample inputs...")
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_ID, subfolder="tokenizer", trust_remote_code=True
    )

    prompt = (
        "A highly detailed photograph of a fluffy orange tabby cat sitting on a "
        "rustic wooden windowsill, gazing out through rain-streaked glass at a "
        "lush green garden filled with blooming roses and lavender bushes. Soft "
        "morning light filters through sheer white curtains, casting warm golden "
        "highlights on the cat's fur. The scene is peaceful and contemplative, "
        "with dewdrops visible on the window pane and a small ceramic vase of "
        "wildflowers placed beside the cat. The background shows rolling hills "
        "under a partly cloudy sky with dramatic crepuscular rays breaking through "
        "the clouds. Shot with a shallow depth of field on a vintage film camera, "
        "giving the image a nostalgic, dreamlike quality with subtle film grain "
        "and slightly desaturated earth tones."
    )
    tokens = tokenizer(
        prompt,
        padding="max_length",
        max_length=SEQ_LEN,
        truncation=True,
        return_tensors="pt",
    )

    input_ids = tokens["input_ids"]  # [1, 512] INT64
    attention_mask = tokens["attention_mask"].to(DTYPE)  # [1, 512] BF16

    # Cache for next time
    torch.save(
        {"input_ids": input_ids, "attention_mask": attention_mask},
        input_cache,
    )

    return input_ids, attention_mask


class TextEncoderPT:
    """Convenience wrapper that loads the model and provides ref output."""

    def __init__(self):
        self.model = _load_text_encoder()
        self.model.eval()

    def __call__(self, input_ids, attention_mask):
        with torch.inference_mode():
            return self.model(input_ids, attention_mask)

    def state_dict_for_ttnn(self):
        return self.model.state_dict_for_ttnn()
