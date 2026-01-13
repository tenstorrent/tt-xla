# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import os

import torch
import torch_xla
import torch_xla.runtime as xr
from deepseek_v3p2_exp_model import ModelArgs, Transformer


# --------------------------------
# Test run
# --------------------------------
def deepseek_test():
    # Set device type
    xr.set_device_type("TT")

    # Create model args with a single layer for testing
    args = ModelArgs(
        n_layers=1,
        q_lora_rank=3072,
    )

    # Instantiate model with precomputed freqs_cis
    # Option 1: Load from file (recommended for performance)
    freqs_cis_path = "/localdev/gengelage/tt-xla/examples/pytorch/DeepSeek_params/freqs_cis_test_config_real.pt"
    model = Transformer(args, freqs_cis_path=freqs_cis_path)

    # Option 2: Compute on the fly (fallback)
    # model = Transformer(args)

    model = model.to(torch.bfloat16)

    # Put it in inference mode and compile it
    model = model.eval()
    compiled_model = torch.compile(model, backend="tt")

    # Create batch of random token IDs
    batch_size = 1
    seq_len = 32
    tokens = torch.randint(0, args.vocab_size, (batch_size, seq_len))

    # Move inputs and model to device
    device = torch_xla.device()
    tokens = tokens.to(device)
    compiled_model = compiled_model.to(device)

    # Run model
    with torch.no_grad():
        output = compiled_model(tokens)


# --------------------------------
# main
# --------------------------------
if __name__ == "__main__":
    deepseek_test()
