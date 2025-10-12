# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
import gc
from transformers import (
    Qwen2_5_VLForConditionalGeneration,
    AutoTokenizer,
    AutoProcessor,
    Qwen2_5_VLConfig,
)
from qwen_vl_utils import process_vision_info
import pytest
from loguru import logger
import torch


variants = [
    "Qwen/Qwen2.5-VL-3B-Instruct",
]


@pytest.mark.parametrize("variant", variants)
def test_qwen_vl(variant):

    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        variant, torch_dtype=torch.float32, low_cpu_mem_usage=True
    )
    model.config.use_cache = False
    model.config.return_dict = False
    min_pixels = 56 * 56
    max_pixels = 13 * 28 * 1280
    processor = AutoProcessor.from_pretrained(
        variant, min_pixels=min_pixels, max_pixels=max_pixels
    )

    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "image": "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-VL/assets/demo.jpeg",
                },
                {"type": "text", "text": "Describe this image."},
            ],
        }
    ]

    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    image_inputs, video_inputs = process_vision_info(messages)

    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )

    # ==== single iter

    op = model(**inputs)

    logger.info("single iter op={}", op)
