# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
import gc
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
import pytest
from loguru import logger
import torch
from collections import Counter


variants = [
    "Qwen/Qwen2.5-VL-3B-Instruct",
]


@pytest.mark.parametrize("variant", variants)
def test_qwen_vl(variant):

    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        variant, torch_dtype=torch.float32, low_cpu_mem_usage=True
    )
    # processor = AutoProcessor.from_pretrained(variant)

    # model.config.return_dict = False
    # model.config.use_cache = False

    # default

    # min_pixels (`int`, *optional*, defaults to `56 * 56`):
    # max_pixels (`int`, *optional*, defaults to `28 * 28 * 1280`):

    min_pixels = 56 * 56
    max_pixels = 13 * 28 * 1280
    processor = AutoProcessor.from_pretrained(
        variant, min_pixels=min_pixels, max_pixels=max_pixels
    )

    logger.info("min_pixels set ={}", min_pixels)
    logger.info("max_pixels set ={}", max_pixels)
    logger.info(
        "processor.image_processor.min_pixels={}", processor.image_processor.min_pixels
    )
    logger.info(
        "processor.image_processor.max_pixels={}", processor.image_processor.max_pixels
    )

    # ==================

    # Header
    logger.info(f"\n{'Type':10s} | {'Layer Name':60s} | {'DType':10s} | {'Device'}")
    logger.info("-" * 95)

    # Track dtypes and devices
    all_dtypes, all_devices = [], []

    # Parameters
    for name, param in model.named_parameters():
        logger.info(
            f"{'PARAM':10s} | {name:60s} | {str(param.dtype):10s} | {param.device}"
        )
        all_dtypes.append(str(param.dtype))
        all_devices.append(str(param.device))

    # Buffers
    for name, buf in model.named_buffers():
        logger.info(
            f"{'BUFFER':10s} | {name:60s} | {str(buf.dtype):10s} | {buf.device}"
        )
        all_dtypes.append(str(buf.dtype))
        all_devices.append(str(buf.device))

    # Summary
    logger.info("-" * 95)
    logger.info(f"DType summary: {dict(Counter(all_dtypes))}")
    logger.info(f"Device summary: {dict(Counter(all_devices))}")

    # ==================

    logger.info("model={}", model)

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

    logger.info("image_inputs={}", image_inputs)
    logger.info("video_inputs={}", video_inputs)

    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )

    logger.info("inputs={}", inputs)

    for name, tensor in inputs.items():
        logger.info(
            "{:<15s} | shape: {} | dtype: {}", name, tuple(tensor.shape), tensor.dtype
        )

    # ==== single iter

    op = model(**inputs)

    logger.info("single iter op={}", op)

    # Inference: Generation of the output
    generated_ids = model.generate(**inputs, max_new_tokens=128)
    generated_ids_trimmed = [
        out_ids[len(in_ids) :]
        for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_text = processor.batch_decode(
        generated_ids_trimmed,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )
    print(output_text)
