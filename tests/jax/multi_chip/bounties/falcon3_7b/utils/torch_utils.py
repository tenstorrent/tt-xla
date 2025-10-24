# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def init_torch_model(model_name: str, config):
    """
    Initialize the PyTorch model with the given configuration.
    """
    torch_model = AutoModelForCausalLM.from_pretrained(
        model_name,
        config=config,
        device_map="cpu",
        torch_dtype=torch.float32,
    )
    return torch_model


def prepare_torch_input(model_name, prompt):
    """
    Prepare input for the PyTorch model.
    """
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    inputs = tokenizer(prompt, return_tensors="pt")

    return tokenizer, inputs.input_ids, inputs.attention_mask


def run_torch_model(torch_model, input_ids, attention_mask):
    """
    Run the PyTorch model with the given input IDs and attention mask.
    """
    print("🏢 Generating HF Model output...")
    outputs = torch_model.generate(
        input_ids=input_ids,
        attention_mask=attention_mask,
    )
    return outputs
