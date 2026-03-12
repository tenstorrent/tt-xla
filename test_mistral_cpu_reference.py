# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Quick CPU reference run for Mistral-Small multimodal to get expected output.

Usage:
    python test_mistral_cpu_reference.py                                          # default: 3.1
    python test_mistral_cpu_reference.py mistralai/Mistral-Small-3.2-24B-Instruct-2506
"""

import sys
import time

import torch
from huggingface_hub import hf_hub_download
from mistral_common.protocol.instruct.messages import ImageChunk, TextChunk, UserMessage
from mistral_common.protocol.instruct.request import ChatCompletionRequest
from mistral_common.tokens.tokenizers.mistral import MistralTokenizer
from PIL import Image
from transformers import AutoModelForImageTextToText

model_name = (
    sys.argv[1]
    if len(sys.argv) > 1
    else "mistralai/Mistral-Small-3.1-24B-Instruct-2503"
)
image_path = "tests/integrations/vllm_plugin/generative/assets/test_battle_scene.png"

image = Image.open(image_path)

print(f"Tokenizing with Mistral tokenizer...")
tokenizer_path = hf_hub_download(model_name, "tekken.json")
tokenizer = MistralTokenizer.from_file(tokenizer_path)
tokenized = tokenizer.encode_chat_completion(
    ChatCompletionRequest(
        messages=[
            UserMessage(
                content=[
                    ImageChunk(image=image),
                    TextChunk(
                        text="What action do you think I should take in this situation? "
                    ),
                ]
            ),
        ],
    )
)

input_ids = torch.tensor([tokenized.tokens], dtype=torch.long)
# Image is preprocessed by the tokenizer into (C, H, W) numpy array per image;
# torch.tensor(list of arrays) gives (N, C, H, W) which is what the model expects.
pixel_values = torch.tensor(tokenized.images).to(torch.bfloat16)
# image_sizes tells the model the (H, W) of each image
image_sizes = torch.tensor([[pv.shape[-2], pv.shape[-1]] for pv in tokenized.images])
print(
    f"Prompt length: {input_ids.shape[1]} tokens, pixel_values: {pixel_values.shape}, image_sizes: {image_sizes}"
)

print(f"Loading model (bfloat16, CPU) — this will take a few minutes...")
t0 = time.time()
model = AutoModelForImageTextToText.from_pretrained(
    model_name, dtype="bfloat16", device_map="cpu"
)
print(f"Model loaded in {time.time() - t0:.1f}s")

num_tokens = 128
print(f"Generating {num_tokens} tokens on CPU — this will be slow...")
t0 = time.time()
output = model.generate(
    input_ids=input_ids,
    pixel_values=pixel_values,
    image_sizes=image_sizes,
    max_new_tokens=num_tokens,
    do_sample=False,
)
elapsed = time.time() - t0
print(f"Generation took {elapsed:.1f}s ({elapsed/num_tokens:.1f}s/token)")

generated_ids = output[0][input_ids.shape[1] :]
generated_text = tokenizer.decode(generated_ids.tolist())
print(f"\n=== Generated only ===\n{generated_text}")

full_text = tokenizer.decode(output[0].tolist())
print(f"\n=== Full output ===\n{full_text}")
