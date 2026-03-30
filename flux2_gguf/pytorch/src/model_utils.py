# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Helper functions for loading GGUF-quantized FLUX.2 models.
"""

import torch
from diffusers.models import Flux2Transformer2DModel
from diffusers import GGUFQuantizationConfig
from huggingface_hub import hf_hub_download


def load_flux2_gguf_transformer(repo_id: str, gguf_filename: str):
    """Load a FLUX.2 transformer from a GGUF-quantized checkpoint.

    Args:
        repo_id: HuggingFace repository ID containing the GGUF file.
        gguf_filename: Filename of the GGUF checkpoint within the repo.

    Returns:
        Flux2Transformer2DModel: Loaded GGUF-quantized transformer model.
    """
    model_path = hf_hub_download(repo_id=repo_id, filename=gguf_filename)

    quantization_config = GGUFQuantizationConfig(compute_dtype=torch.bfloat16)

    transformer = Flux2Transformer2DModel.from_single_file(
        model_path,
        quantization_config=quantization_config,
        torch_dtype=torch.bfloat16,
    )

    transformer.eval()
    for param in transformer.parameters():
        if param.requires_grad:
            param.requires_grad = False

    return transformer
