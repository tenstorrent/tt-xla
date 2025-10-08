# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
import os

import numpy as np
import torch
from torch import autocast
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig


def run():
    model_id = "meta-llama/Meta-Llama-3.1-8B"
    prompt = "What is the name of the largest planet in our solar system?"
    max_gen_len = 64
    temperature = 0.0
    top_p = 1.0
    max_seq_len = 128
    n_layers = 32

    print("📦 Loading model and tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    tokenizer.pad_token_id = tokenizer.eos_token_id

    # Load full model first
    model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.float32)

    # ✅ Truncate to first 16 layers
    model.config.max_position_embeddings = max_seq_len
    model.model.layers = torch.nn.ModuleList(list(model.model.layers)[:n_layers])
    model.config.num_hidden_layers = n_layers
    model.base_model.config.num_hidden_layers = n_layers

    model.eval()
    model = model.to("cuda" if torch.cuda.is_available() else "cpu")

    print("✍️ Tokenizing prompt...")
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    print(model.config)
    print("⚙️ Generating...")
    generation_config = GenerationConfig(
        max_length=max_gen_len + len(inputs["input_ids"][0]),
        temperature=temperature,
        top_p=top_p,
        do_sample=False,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
        num_beams=1,
    )

    with torch.no_grad():
        output = model.generate(**inputs, generation_config=generation_config)

    if not os.path.isdir("results"):
        os.mkdir("results")

    np.savetxt("results/hf.txt", output, fmt="%d")
    decoded = tokenizer.decode(output[0], skip_special_tokens=False)

    print("\n🧠 Output:")
    print(decoded)


if __name__ == "__main__":
    run()
