# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
Simplified CLIP encoder layer for debugging metadata propagation.
"""

import torch
import torch.nn as nn
import torch_xla
import torch_xla.runtime as xr
from transformers import CLIPVisionModelWithProjection
from tt_torch import codegen_py

# CONFIG
compile_options = {
    "optimization_level": 1,
    "codegen_try_recover_structure": True,
}
EXPORT_PATH = "clip_single_layer_codegen"
torch_xla.set_custom_compile_options(compile_options)

dtype = torch.bfloat16
image_encoder_id = "laion/CLIP-ViT-H-14-laion2B-s32B-b79K"


class SingleCLIPEncoderLayer(nn.Module):
    """Just a single CLIP encoder layer for debugging."""

    def __init__(self, encoder_layer):
        super().__init__()
        self.encoder_layer = encoder_layer

    def forward(self, hidden_states):
        # CLIPEncoderLayer expects (hidden_states, attention_mask, causal_attention_mask, output_attentions)
        # We pass None for masks and False for output_attentions
        outputs = self.encoder_layer(hidden_states, None, None, False)
        return outputs[0]  # Just the hidden states


def get_model():
    """Load just a single encoder layer from CLIP."""
    print(f"Loading CLIP Vision Encoder in {dtype}...")
    image_encoder = CLIPVisionModelWithProjection.from_pretrained(
        image_encoder_id, torch_dtype=dtype
    )

    # Extract just one encoder layer (e.g., layer 0)
    single_layer = image_encoder.vision_model.encoder.layers[0]

    model = SingleCLIPEncoderLayer(single_layer)
    model.eval()
    return model


def get_input():
    """Create a dummy input matching the shape after embeddings.

    CLIP ViT-H has:
    - hidden_size = 1280
    - num_patches = 16*16 = 256 (for 224x224 image with 14x14 patches)
    - +1 for CLS token = 257 tokens
    """
    batch_size = 1
    seq_len = 257  # 256 patches + 1 CLS token
    hidden_size = 1280
    return torch.randn(batch_size, seq_len, hidden_size, dtype=dtype)


def run_codegen():
    """Generate Python code for a single encoder layer."""
    xr.set_device_type("TT")

    model = get_model()
    input_tensor = get_input()

    print(f"Input shape: {input_tensor.shape}")
    print(f"Generating code to {EXPORT_PATH}...")

    codegen_py(
        model,
        input_tensor,
        export_path=EXPORT_PATH,
        compiler_options=compile_options,
    )
    print(f"Code generated to {EXPORT_PATH}")


def main():
    run_codegen()


if __name__ == "__main__":
    main()
