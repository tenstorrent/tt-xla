# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
Code adapted from:
- https://huggingface.co/deepseek-ai/DeepSeek-OCR/blob/main/conversation.py
- https://huggingface.co/deepseek-ai/DeepSeek-OCR/blob/main/modeling_deepseekocr.py
"""

from typing import List, Optional, Tuple
from PIL import Image, ImageOps
import torch
import torch.nn as nn
from torchvision import transforms
from addict import Dict
from abc import ABC
import math
import dataclasses
from enum import IntEnum, auto
from typing import Dict as TypingDict


class SeparatorStyle(IntEnum):
    PLAIN = auto()


@dataclasses.dataclass
class Conversation:

    name: str
    system_template: str = "{system_message}"
    system_message: str = ""
    roles: List[str] = (("USER", "ASSISTANT"),)
    messages: List[List[str]] = ()
    offset: int = 0
    sep_style: SeparatorStyle = SeparatorStyle.PLAIN
    sep: str = "\n"
    sep2: str = None
    stop_str: str = None
    stop_token_ids: List[int] = None

    def get_prompt(self) -> str:
        if self.sep_style == SeparatorStyle.PLAIN:
            seps = [self.sep, self.sep2]
            ret = ""
            for i, (role, message) in enumerate(self.messages):
                if message:
                    if type(message) is tuple:
                        message, _, _ = message
                    if i % 2 == 0:
                        ret += message + seps[i % 2]
                    else:
                        ret += message + seps[i % 2]
                else:
                    ret += ""
            return ret
        else:
            raise ValueError(f"Invalid style: {self.sep_style}")

    def set_system_message(self, system_message: str):
        self.system_message = system_message

    def append_message(self, role: str, message: str):
        self.messages.append([role, message])

    def copy(self):
        return Conversation(
            name=self.name,
            system_template=self.system_template,
            system_message=self.system_message,
            roles=self.roles,
            messages=[[x, y] for x, y in self.messages],
            offset=self.offset,
            sep_style=self.sep_style,
            sep=self.sep,
            sep2=self.sep2,
            stop_str=self.stop_str,
            stop_token_ids=self.stop_token_ids,
        )


conv_templates: TypingDict[str, Conversation] = {}


def register_conv_template(template: Conversation, override: bool = False):
    if not override:
        assert (
            template.name not in conv_templates
        ), f"{template.name} has been registered."

    conv_templates[template.name] = template


def get_conv_template(name: str) -> Conversation:
    return conv_templates[name].copy()


register_conv_template(
    Conversation(
        name="plain",
        system_template="",
        system_message="",
        roles=("", ""),
        messages=(),
        offset=0,
        sep_style=SeparatorStyle.PLAIN,
        sep="",
        sep2="",
        stop_token_ids=[100001],
        stop_str=["</s>"],
    )
)


def load_image(image_path):

    try:
        image = Image.open(image_path)
        corrected_image = ImageOps.exif_transpose(image)
        return corrected_image

    except Exception as e:
        print(f"error: {e}")
        try:
            return Image.open(image_path)
        except:
            return None


def format_messages(
    conversations: List[Dict[str, str]],
    sft_format: str = "deepseek",
    system_prompt: str = "",
):
    conv = get_conv_template(sft_format)
    conv.set_system_message(system_prompt)
    for message in conversations:
        conv.append_message(message["role"], message["content"].strip())
    sft_prompt = conv.get_prompt().strip()

    return sft_prompt


def load_pil_images(conversations: List[Dict[str, str]]) -> List[Image.Image]:

    pil_images = []

    for message in conversations:
        if "images" not in message:
            continue

        for image_path in message["images"]:
            pil_img = load_image(image_path)
            pil_img = pil_img.convert("RGB")
            pil_images.append(pil_img)

    return pil_images


def normalize_transform(mean, std):
    if mean is None and std is None:
        transform = None
    elif mean is None and std is not None:
        mean = [0.0] * len(std)
        transform = transforms.Normalize(mean=mean, std=std)
    elif mean is not None and std is None:
        std = [1.0] * len(mean)
        transform = transforms.Normalize(mean=mean, std=std)
    else:
        transform = transforms.Normalize(mean=mean, std=std)

    return transform


class BaseTransform(ABC):
    def __call__(self, *args, **kwargs) -> torch.Tensor:
        pass

    @property
    def default_shape(self):
        raise NotImplementedError


class BasicImageTransform(BaseTransform):
    def __init__(
        self,
        mean: Optional[Tuple[float, float, float]] = (0.5, 0.5, 0.5),
        std: Optional[Tuple[float, float, float]] = (0.5, 0.5, 0.5),
        normalize: bool = True,
    ):
        self.mean = mean
        self.std = std

        transform_pipelines = [transforms.ToTensor()]

        normalize = normalize_transform(mean, std) if normalize else nn.Identity()
        if normalize is not None:
            transform_pipelines.append(normalize)

        self.transform = transforms.Compose(transform_pipelines)

    def __call__(self, x):
        x = self.transform(x)
        return x


def text_encode(tokenizer, text: str, bos: bool = True, eos: bool = False):
    t = tokenizer.encode(text, add_special_tokens=False)
    bos_id = 0
    eos_id = 1
    if bos:
        t = [bos_id] + t
    if eos:
        t = t + [eos_id]

    return t


def find_closest_aspect_ratio(aspect_ratio, target_ratios, width, height, image_size):
    best_ratio_diff = float("inf")
    best_ratio = (1, 1)
    area = width * height
    for ratio in target_ratios:
        target_aspect_ratio = ratio[0] / ratio[1]
        ratio_diff = abs(aspect_ratio - target_aspect_ratio)
        if ratio_diff < best_ratio_diff:
            best_ratio_diff = ratio_diff
            best_ratio = ratio
        elif ratio_diff == best_ratio_diff:
            if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
                best_ratio = ratio
    return best_ratio


def dynamic_preprocess(
    image, min_num=2, max_num=9, image_size=640, use_thumbnail=False
):
    orig_width, orig_height = image.size
    aspect_ratio = orig_width / orig_height
    target_ratios = set(
        (i, j)
        for n in range(min_num, max_num + 1)
        for i in range(1, n + 1)
        for j in range(1, n + 1)
        if i * j <= max_num and i * j >= min_num
    )

    target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])

    target_aspect_ratio = find_closest_aspect_ratio(
        aspect_ratio, target_ratios, orig_width, orig_height, image_size
    )

    target_width = image_size * target_aspect_ratio[0]
    target_height = image_size * target_aspect_ratio[1]
    blocks = target_aspect_ratio[0] * target_aspect_ratio[1]

    resized_img = image.resize((target_width, target_height))
    processed_images = []
    for i in range(blocks):
        box = (
            (i % (target_width // image_size)) * image_size,
            (i // (target_width // image_size)) * image_size,
            ((i % (target_width // image_size)) + 1) * image_size,
            ((i // (target_width // image_size)) + 1) * image_size,
        )
        split_img = resized_img.crop(box)
        processed_images.append(split_img)
    assert len(processed_images) == blocks
    if use_thumbnail and len(processed_images) != 1:
        thumbnail_img = image.resize((image_size, image_size))
        processed_images.append(thumbnail_img)
    return processed_images, target_aspect_ratio


def preprocess(
    tokenizer, prompt, image_file, base_size=1024, image_size=640, crop_mode=True
):

    conversation = [
        {
            "role": "<|User|>",
            "content": f"{prompt}",
            "images": [f"{image_file}"],
        },
        {"role": "<|Assistant|>", "content": ""},
    ]

    prompt = format_messages(
        conversations=conversation, sft_format="plain", system_prompt=""
    )

    patch_size = 16
    downsample_ratio = 4
    images = load_pil_images(conversation)

    valid_img_tokens = 0
    ratio = 1

    image_draw = images[0].copy()
    w, h = image_draw.size
    ratio = 1 - ((max(w, h) - min(w, h)) / (max(w, h)))

    image_transform = BasicImageTransform(
        mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5), normalize=True
    )
    images_seq_mask = []

    image_token = "<image>"
    image_token_id = 128815
    text_splits = prompt.split(image_token)

    images_list, images_crop_list, images_seq_mask = [], [], []
    tokenized_str = []
    images_spatial_crop = []

    for text_sep, image in zip(text_splits, images):

        tokenized_sep = text_encode(tokenizer, text_sep, bos=False, eos=False)
        tokenized_str += tokenized_sep
        images_seq_mask += [False] * len(tokenized_sep)

        if crop_mode:

            if image.size[0] <= 640 and image.size[1] <= 640:
                crop_ratio = [1, 1]

            else:
                if crop_mode:
                    images_crop_raw, crop_ratio = dynamic_preprocess(image)
                else:
                    crop_ratio = [1, 1]

            global_view = ImageOps.pad(
                image,
                (base_size, base_size),
                color=tuple(int(x * 255) for x in image_transform.mean),
            )

            if base_size == 1024:
                valid_img_tokens += int(256 * ratio)
            elif base_size == 1280:
                valid_img_tokens += int(400 * ratio)

            images_list.append(image_transform(global_view))
            width_crop_num, height_crop_num = crop_ratio

            images_spatial_crop.append([width_crop_num, height_crop_num])

            if width_crop_num > 1 or height_crop_num > 1:

                for i in range(len(images_crop_raw)):
                    images_crop_list.append(image_transform(images_crop_raw[i]))

            if image_size == 640:
                valid_img_tokens += len(images_crop_list) * 100

            num_queries = math.ceil((image_size // patch_size) / downsample_ratio)
            num_queries_base = math.ceil((base_size // patch_size) / downsample_ratio)

            tokenized_image = (
                [image_token_id] * num_queries_base + [image_token_id]
            ) * num_queries_base
            tokenized_image += [image_token_id]
            if width_crop_num > 1 or height_crop_num > 1:
                tokenized_image += (
                    [image_token_id] * (num_queries * width_crop_num) + [image_token_id]
                ) * (num_queries * height_crop_num)
            tokenized_str += tokenized_image
            images_seq_mask += [True] * len(tokenized_image)

    tokenized_sep = text_encode(tokenizer, text_splits[-1], bos=False, eos=False)
    tokenized_str += tokenized_sep
    images_seq_mask += [False] * len(tokenized_sep)

    bos_id = 0
    tokenized_str = [bos_id] + tokenized_str
    images_seq_mask = [False] + images_seq_mask

    input_ids = torch.LongTensor(tokenized_str)
    images_seq_mask = torch.tensor(images_seq_mask, dtype=torch.bool)

    if len(images_list) == 0:
        images_ori = torch.zeros((1, 3, image_size, image_size))
        images_spatial_crop = torch.zeros((1, 2), dtype=torch.long)
        images_crop = torch.zeros((1, 3, base_size, base_size))

    else:
        images_ori = torch.stack(images_list, dim=0)
        images_spatial_crop = torch.tensor(images_spatial_crop, dtype=torch.long)
        if images_crop_list:
            images_crop = torch.stack(images_crop_list, dim=0)
        else:
            images_crop = torch.zeros((1, 3, base_size, base_size))

    images = [(images_crop, images_ori)]

    return {
        "input_ids": input_ids.unsqueeze(0),
        "images": images,
        "images_seq_mask": images_seq_mask.unsqueeze(0),
        "images_spatial_crop": images_spatial_crop,
    }
