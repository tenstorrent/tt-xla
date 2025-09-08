# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
import torch

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
