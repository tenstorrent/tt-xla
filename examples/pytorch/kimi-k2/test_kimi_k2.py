# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import os
import sys

import torch
import torch_xla
import torch_xla.runtime as xr

# Add the kimi-k2 directory to the path so we can import the modules
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from configuration_deepseek import DeepseekV3Config
from modeling_deepseek import DeepseekV3ForCausalLM


# --------------------------------
# Test run
# --------------------------------
def kimi_k2_test():
    # Set device type
    xr.set_device_type("TT")

    # Create model config with a single layer for testing
    config = DeepseekV3Config(
        num_hidden_layers=1,  # Single layer for testing
    )

    # Instantiate model
    print("Creating Kimi-K2 model...")
    model = DeepseekV3ForCausalLM(config)

    # Convert to bfloat16
    model = model.to(torch.bfloat16)

    # Put it in inference mode
    model = model.eval()

    print("Compiling model with XLA backend...")
    compiled_model = torch.compile(model, backend="tt")

    # Create batch of random token IDs
    batch_size = 1
    seq_len = 32
    print(f"Creating input tokens: batch_size={batch_size}, seq_len={seq_len}")
    tokens = torch.randint(0, config.vocab_size, (batch_size, seq_len))

    # Move inputs and model to device
    device = torch_xla.device()
    print(f"Moving to device: {device}")
    tokens = tokens.to(device)
    compiled_model = compiled_model.to(device)

    # Run model
    print("Running inference...")
    with torch.no_grad():
        output = compiled_model(tokens)

    print(f"Output shape: {output.logits.shape}")
    print("Test completed successfully!")


# --------------------------------
# main
# --------------------------------
if __name__ == "__main__":
    kimi_k2_test()
