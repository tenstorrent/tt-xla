# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
Code adapted from: https://github.com/X-PLUG/mPLUG-Owl/tree/main/mPLUG-Owl2
License: https://github.com/X-PLUG/mPLUG-Owl/blob/main/LICENSE
"""

import torch
from loguru import logger
from tqdm import tqdm
import json
import gc
import os
from ....tools.utils import get_file


DEFAULT_IMAGE_TOKEN = "<|image|>"
IMAGE_TOKEN_INDEX = -200
IGNORE_INDEX = -100


def process_images(images, image_processor):

    new_images = []

    for image in images:
        max_edge = max(image.size)
        image = image.resize((max_edge, max_edge))
        image = image_processor.preprocess(image, return_tensors="pt")["pixel_values"][
            0
        ]
        new_images.append(image)

    if all(x.shape == new_images[0].shape for x in new_images):
        new_images = torch.stack(new_images, dim=0)

    return new_images


def tokenizer_image_token(
    prompt, tokenizer, image_token_index=IMAGE_TOKEN_INDEX, return_tensors=None
):
    prompt_chunks = [
        tokenizer(chunk).input_ids if len(chunk) > 0 else []
        for chunk in prompt.split(DEFAULT_IMAGE_TOKEN)
    ]

    def insert_separator(X, sep):
        return [ele for sublist in zip(X, [sep] * len(X)) for ele in sublist][:-1]

    input_ids = []
    offset = 0
    if (
        len(prompt_chunks) > 0
        and len(prompt_chunks[0]) > 0
        and prompt_chunks[0][0] == tokenizer.bos_token_id
    ):
        offset = 1
        input_ids.append(prompt_chunks[0][0])

    for x in insert_separator(prompt_chunks, [image_token_index] * (offset + 1)):
        input_ids.extend(x[offset:])

    if return_tensors is not None:
        if return_tensors == "pt":
            return torch.tensor(input_ids, dtype=torch.long)
        raise ValueError(f"Unsupported tensor type: {return_tensors}")

    return input_ids


def load_weights(model):

    # Download the index file first
    weights_json = get_file(
        "test_files/pytorch/mplug_owl2/pytorch_model.bin.index.json"
    )

    # Download all 33 weight files
    logger.info("Downloading model weight files...")
    weight_files = {}
    for i in range(1, 34):
        filename = f"pytorch_model-{i}-of-33.bin"
        file_path = get_file(f"test_files/pytorch/mplug_owl2/{filename}")
        weight_files[filename] = file_path
        logger.info(f"Downloaded {filename}")

    # Get the directory path from any of the downloaded files
    weights_dir = os.path.dirname(list(weight_files.values())[0])

    with open(weights_json, "r") as f:
        index = json.load(f)

    shard_files = list(set(index["weight_map"].values()))
    expected_keys = set(index["weight_map"].keys())

    for shard_file in tqdm(shard_files, desc="Loading checkpoint shards"):
        shard_path = os.path.join(weights_dir, shard_file)
        if not os.path.exists(shard_path):
            logger.error(f"Weight file not found: {shard_path}")
            continue

        state_dict = torch.load(shard_path, map_location="cpu")
        model.load_state_dict(state_dict, strict=False)

        del state_dict
        gc.collect()

    model_keys = set(model.state_dict().keys())
    missing_keys = model_keys - expected_keys
    unexpected_keys = expected_keys - model_keys

    if missing_keys:
        logger.info("Missing keys: {}", list(missing_keys))

    if unexpected_keys:
        logger.info("Unexpected keys: {}", list(unexpected_keys))

    logger.info("Model loaded successfully!")
    return model
