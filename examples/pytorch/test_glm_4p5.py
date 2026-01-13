# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
Test script for GLM-4.5 model.

Usage:
    python test_glm_4p5.py
"""

import torch
import torch_xla
import torch_xla.runtime as xr
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer


def test_glm_4p5():
    """Test GLM-4.5 model with simple random tokens (alternative approach)."""
    # Set device type
    xr.set_device_type("TT")

    print("Loading GLM-4.5 config from transformers...")
    # Load config from pretrained model
    config = AutoConfig.from_pretrained("zai-org/GLM-4.5", trust_remote_code=True)

    # Modify config for single layer testing
    print(f"Original num_hidden_layers: {config.num_hidden_layers}")
    config.num_hidden_layers = 1
    print(f"Modified num_hidden_layers: {config.num_hidden_layers}")

    # Instantiate model with modified config
    print("\nCreating GLM-4.5 model with modified config...")
    model = AutoModelForCausalLM.from_config(config, trust_remote_code=True)

    # Convert to bfloat16
    model = model.to(torch.bfloat16)

    # Put it in inference mode
    model = model.eval()

    print("\nCompiling model with XLA backend...")
    compiled_model = torch.compile(model, backend="tt")

    # Create batch of random token IDs
    batch_size = 1
    seq_len = 32
    print(f"\nCreating random input tokens: batch_size={batch_size}, seq_len={seq_len}")

    # Generate random tokens (avoiding potential special tokens at the high end)
    # GLM-4.5 vocab_size is around 151,000+, use safe range
    tokens = torch.randint(
        0, min(config.vocab_size - 1000, 150000), (batch_size, seq_len)
    )

    # Move inputs and model to device
    device = torch_xla.device()
    print(f"Moving to device: {device}")
    tokens = tokens.to(device)
    compiled_model = compiled_model.to(device)

    # Run model
    print("\nRunning inference...")
    with torch.no_grad():
        output = compiled_model(tokens)

    print(f"Output logits shape: {output.logits.shape}")
    print("\nGLM-4.5 simple test completed successfully!")


# --------------------------------
# main
# --------------------------------
if __name__ == "__main__":
    test_glm_4p5()
