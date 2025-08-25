# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
import torch
from transformers import AutoConfig
from utils.torch_utils import *

MODEL_NAME = "tiiuae/Falcon3-7B-Instruct"
EXAMPLE_PROMPT = """
Q: Janet's ducks lay 16 eggs per day. She eats three for breakfast every morning and bakes muffins for her friends every day with four.
She sells the remainder at the farmers' market daily for $2 per fresh duck egg.
How much in dollars does she make every day at the farmers' market?\n
A: """


def main(model_name: str, prompt: str):
    """
    Run the test comparing Hugging Face and Flax models.
    """
    print("🪄  Initializing models...")
    config = AutoConfig.from_pretrained(
        model_name,
        num_hidden_layers=28,  # set smaller for easeier testing
        torch_dtype=torch.float32,
    )
    tokenizer, input_ids, attention_mask = prepare_torch_input(model_name, prompt)

    torch_model = init_torch_model(model_name, config)
    torch_output = run_torch_model(torch_model, input_ids, attention_mask)

    torch_result = tokenizer.batch_decode(torch_output, skip_special_tokens=False)
    print("🈵 Decoded output:", torch_result[0])


if __name__ == "__main__":
    torch_output = main(model_name=MODEL_NAME, prompt=EXAMPLE_PROMPT)
