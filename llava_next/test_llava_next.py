# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import requests
import torch
import torch_xla.core.xla_model as xm
import torch_xla.runtime as xr
import tt_torch
from loguru import logger
from PIL import Image
from transformers import LlavaNextForConditionalGeneration, LlavaNextProcessor


def test_llava_next():
    # Set the XLA runtime device to TT
    xr.set_device_type("TT")

    # Load processor and model
    processor = LlavaNextProcessor.from_pretrained("llava-hf/llama3-llava-next-8b-hf")
    model = LlavaNextForConditionalGeneration.from_pretrained(
        "llava-hf/llama3-llava-next-8b-hf", dtype=torch.bfloat16, return_dict=False
    )
    model.eval()

    # Prepare image and text prompt
    url = "https://github.com/haotian-liu/LLaVA/blob/1a91fc274d7c35a9b50b3cb29c4247ae5837ce39/images/llava_v1_5_radar.jpg?raw=true"
    image = Image.open(requests.get(url, stream=True).raw)

    conversation = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "What is shown in this image?"},
                {"type": "image"},
            ],
        },
    ]
    prompt = processor.apply_chat_template(conversation, add_generation_prompt=True)
    inputs = processor(images=image, text=prompt, return_tensors="pt")
    inputs["pixel_values"] = inputs["pixel_values"].to(torch.bfloat16)

    # Compile the model using XLA with TT backend
    compiled_model = torch.compile(model, backend="tt")

    # Move model and inputs to TT device
    device = xm.xla_device()
    compiled_model = compiled_model.to(device)
    inputs = {k: v.to(device) for k, v in inputs.items()}

    # Run inference on Tenstorrent device
    with torch.no_grad():
        outputs = compiled_model(**inputs, use_cache=False)
