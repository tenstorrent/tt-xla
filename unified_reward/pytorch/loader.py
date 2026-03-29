# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
UnifiedReward model loader implementation for multimodal reward scoring.
"""

import copy

import torch
from PIL import Image
from typing import Optional

from llava.constants import IMAGE_TOKEN_INDEX
from llava.conversation import conv_templates
from llava.mm_utils import process_images, tokenizer_image_token
from llava.model.builder import load_pretrained_model

from ...base import ForgeModel
from ...config import (
    ModelConfig,
    ModelInfo,
    ModelGroup,
    ModelTask,
    ModelSource,
    Framework,
    StrEnum,
)
from ...tools.utils import get_file


class ModelVariant(StrEnum):
    """Available UnifiedReward model variants."""

    UNIFIED_REWARD_7B_V1_5 = "7b_v1.5"


class ModelLoader(ForgeModel):
    """UnifiedReward model loader for multimodal reward scoring."""

    _VARIANTS = {
        ModelVariant.UNIFIED_REWARD_7B_V1_5: ModelConfig(
            pretrained_model_name="CodeGoat24/UnifiedReward-7b-v1.5",
        ),
    }

    DEFAULT_VARIANT = ModelVariant.UNIFIED_REWARD_7B_V1_5

    def __init__(self, variant: Optional[ModelVariant] = None):
        """Initialize UnifiedReward model loader."""
        super().__init__(variant)
        self.tokenizer = None
        self.image_processor = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT
        return ModelInfo(
            model="UnifiedReward",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.MM_CONDITIONAL_GENERATION,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def load_model(self, *, dtype_override=None, **kwargs):
        """Load and return the UnifiedReward model instance."""
        model_name_or_path = self._variant_config.pretrained_model_name
        tokenizer, model, image_processor, _ = load_pretrained_model(
            model_name_or_path, None, "llava_qwen", device_map="cpu"
        )
        model.eval()

        if dtype_override:
            model = model.to(dtype_override)

        self.tokenizer = tokenizer
        self.image_processor = image_processor

        return model

    def load_inputs(self, dtype_override=None, batch_size=1):
        """Load and return input tensors for UnifiedReward."""
        if self.tokenizer is None or self.image_processor is None:
            raise RuntimeError("load_model() must be called before load_inputs()")

        image_file = get_file("http://images.cocodataset.org/val2017/000000039769.jpg")
        image = Image.open(image_file)

        image_tensor = process_images([image], self.image_processor, None)
        if dtype_override:
            image_tensor = [img.to(dtype_override) for img in image_tensor]

        critic_prompt = (
            "Given an image and a corresponding question, please serve as an unbiased "
            "and fair judge to evaluate the quality of answer answers provided by a "
            "Large Multimodal Model (LMM). Score the response out of 100 and explain "
            "your reasoning with specific details. Your task is provided as follows:\n"
            "Question: [What are these?]\n"
            "The LMM response: [The image shows two cats lying on a pink blanket.]\n"
            "ASSISTANT:\n"
        )

        from llava.constants import DEFAULT_IMAGE_TOKEN

        question = DEFAULT_IMAGE_TOKEN + "\n" + critic_prompt
        conv = copy.deepcopy(conv_templates["qwen_1_5"])
        conv.append_message(conv.roles[0], question)
        conv.append_message(conv.roles[1], None)
        prompt_question = conv.get_prompt()

        input_ids = tokenizer_image_token(
            prompt_question, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt"
        ).unsqueeze(0)

        image_sizes = [image.size]

        return {
            "input_ids": input_ids,
            "images": image_tensor,
            "image_sizes": image_sizes,
        }

    def decode_output(self, outputs, **kwargs):
        """Decode model outputs into human-readable text."""
        if self.tokenizer is None:
            raise RuntimeError("load_model() must be called before decode_output()")

        if isinstance(outputs, str):
            return outputs

        if torch.is_tensor(outputs) and outputs.dtype in [torch.long, torch.int]:
            return self.tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]

        logits = outputs.logits if hasattr(outputs, "logits") else outputs[0]
        next_token_id = torch.argmax(logits[:, -1, :], dim=-1)
        return self.tokenizer.decode(next_token_id)
