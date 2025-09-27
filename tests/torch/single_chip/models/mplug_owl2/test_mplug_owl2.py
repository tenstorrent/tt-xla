# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
from third_party.tt_forge_models.tools.utils import get_file
import torch
from transformers import AutoTokenizer
from tests.torch.single_chip.models.mplug_owl2.src.modeling_mplug_owl2 import (
    MPLUGOwl2LlamaForCausalLM,
)
from transformers.models.clip.image_processing_clip import CLIPImageProcessor
from tests.torch.single_chip.models.mplug_owl2.src.conversation import conv_templates
from tests.torch.single_chip.models.mplug_owl2.src.model_utils import (
    process_images,
    DEFAULT_IMAGE_TOKEN,
    IMAGE_TOKEN_INDEX,
    tokenizer_image_token,
)
from PIL import Image
from loguru import logger


def test_mplug_owl2():

    # model
    model_path = "MAGAer13/mplug-owl2-llama2-7b"
    tokenizer = AutoTokenizer.from_pretrained(
        model_path, use_fast=False, trust_remote_code=True
    )
    model = MPLUGOwl2LlamaForCausalLM.from_pretrained(
        model_path, return_dict=False, use_cache=False, torch_dtype=torch.float32
    )
    image_processor = CLIPImageProcessor.from_pretrained(model_path)

    logger.info("model={}", model)

    # inputs
    query = "Describe the image."
    image_file = get_file("http://images.cocodataset.org/val2017/000000039769.jpg")

    # prepare inputs
    conv = conv_templates["mplug_owl2"].copy()
    image = Image.open(image_file).convert("RGB")
    max_edge = max(image.size)
    image = image.resize((max_edge, max_edge))
    image_tensor = process_images([image], image_processor)

    inp = DEFAULT_IMAGE_TOKEN + query
    conv.append_message(conv.roles[0], inp)
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt()

    input_ids = tokenizer_image_token(
        prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt"
    ).unsqueeze(0)

    op = model(input_ids, images=image_tensor)

    print("my op=", op)

    torch.save(op[0], "my_new_logits.pt")
