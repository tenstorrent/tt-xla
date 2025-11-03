# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
from collections import Counter

import pytest
import requests
import torch
from loguru import logger
from PIL import Image
from transformers import MaskFormerFeatureExtractor, MaskFormerForInstanceSegmentation

variants = ["facebook/maskformer-swin-base-coco", "facebook/maskformer-swin-base-ade"]


@pytest.mark.parametrize("variant", variants)
def test_maskformer_swim_b(variant):

    dtype_override = torch.bfloat16
    feature_extractor = MaskFormerFeatureExtractor.from_pretrained(variant)

    model_kwargs = {"return_dict": False}
    if dtype_override is not None:
        model_kwargs["torch_dtype"] = dtype_override

    model = MaskFormerForInstanceSegmentation.from_pretrained(variant, **model_kwargs)
    model.config.return_dict = False

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

    url = "http://images.cocodataset.org/val2017/000000039769.jpg"
    image = Image.open(requests.get(url, stream=True).raw)
    inputs = feature_extractor(images=image, return_tensors="pt")

    # Optional dtype override
    if dtype_override is not None:
        inputs["pixel_values"] = inputs["pixel_values"].to(dtype_override)

    outputs = model(**inputs)

    logger.info("outputs={}", outputs)
