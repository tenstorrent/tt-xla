# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import torch
import torch_xla.core.xla_model as xm
import torch_xla.runtime as xr
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    StaticCache,
)
# --------------------------------
# Test run
# --------------------------------
def llama():
    # Instantiate model.
    model_name:str = "meta-llama/Llama-3.2-3B"
    model: torch.nn.Module = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16)
    model.config.num_hidden_layers = 2

    # Instantiate tokenizer.
    tokenizer:AutoTokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token


    # Put it in inference mode and compile it.
    model = model.eval()
    model.compile(backend="tt")

    '''
    # Generate inputs.
    input = torch.ones((4, 1, 28, 28), dtype=torch.bfloat16)

    # Connect the device.
    device = xm.xla_device()

    # Move inputs and model to device.
    input = input.to(device)
    model = model.to(device)

    # Run model (with no gradient calculation since we only need inference).
    with torch.no_grad():
        output = model(input)

    print(output)
    '''

# --------------------------------
# main
# --------------------------------
if __name__ == "__main__":
    # By default torch_xla uses the CPU device so we have to set it to TT device.
    xr.set_device_type("TT")

    llama()
