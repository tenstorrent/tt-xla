# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
PyTorch reference wrapper for Qwen3 text encoder.

Loads Qwen3Model from Tongyi-MAI/Z-Image-Turbo / text_encoder subfolder
and provides a clean forward() interface that matches the TTNN graph inputs.
"""

import torch

MODEL_ID = "Tongyi-MAI/Z-Image-Turbo"
SEQ_LEN = 7  # dummy caption length used during tracing


def load_model():
    """Load Qwen3 text encoder (base model, no LM head) in bfloat16.

    Returns:
        (tokenizer, text_encoder): AutoTokenizer and Qwen3Model in eval mode.
    """
    from transformers import AutoModel, AutoTokenizer

    print(f"Loading text encoder from {MODEL_ID}/text_encoder ...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, subfolder="tokenizer")
    text_encoder = AutoModel.from_pretrained(
        MODEL_ID,
        subfolder="text_encoder",
        torch_dtype=torch.bfloat16,
        use_cache=False,
    )
    text_encoder.eval()
    print(
        f"  Loaded text encoder ({sum(p.numel() for p in text_encoder.parameters())/1e9:.2f}B params)"
    )
    return tokenizer, text_encoder


def forward(text_encoder, input_ids, attention_mask=None):
    """Run a CPU forward pass through the text encoder.

    Args:
        text_encoder: Qwen3Model in eval mode.
        input_ids:    [seq_len] or [1, seq_len] int64 token ID tensor.
        attention_mask: optional [1, seq_len] bool/int64 attention mask.

    Returns:
        last_hidden_state: [seq_len, 2560] bfloat16 tensor (squeezed batch dim).
    """
    if input_ids.dim() == 1:
        input_ids = input_ids.unsqueeze(0)  # [1, seq_len]

    with torch.no_grad():
        outputs = text_encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )
    last_hidden = outputs.last_hidden_state  # [1, seq_len, 2560]
    return last_hidden.squeeze(0)  # [seq_len, 2560]
