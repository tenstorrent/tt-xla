# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Ovis2 multimodal model loader implementation for image-text-to-text generation.
"""

import torch
from PIL import Image
from transformers import AutoModelForCausalLM
from typing import Optional

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
    """Available Ovis2 model variants."""

    OVIS2_4B = "4B"


class ModelLoader(ForgeModel):
    """Ovis2 multimodal model loader for image-text-to-text generation."""

    _VARIANTS = {
        ModelVariant.OVIS2_4B: ModelConfig(
            pretrained_model_name="AIDC-AI/Ovis2-4B",
        ),
    }

    DEFAULT_VARIANT = ModelVariant.OVIS2_4B

    sample_image = "http://images.cocodataset.org/val2017/000000039769.jpg"
    sample_text = "Describe this image in detail."

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self.text_tokenizer = None
        self.visual_tokenizer = None
        self.model = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        return ModelInfo(
            model="Ovis2",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.NLP_IMAGE_TO_TEXT,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def _load_model_instance(self, dtype_override=None, **kwargs):
        model_name = self._variant_config.pretrained_model_name
        model_kwargs = {
            "trust_remote_code": True,
            "multimodal_max_length": 8192,
        }
        if dtype_override is not None:
            model_kwargs["torch_dtype"] = dtype_override
        else:
            model_kwargs["torch_dtype"] = torch.bfloat16
        model_kwargs |= kwargs

        model = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)
        model.eval()

        self.model = model
        self.text_tokenizer = model.get_text_tokenizer()
        self.visual_tokenizer = model.get_visual_tokenizer()
        return model

    def load_model(self, *, dtype_override=None, **kwargs):
        return self._load_model_instance(dtype_override=dtype_override, **kwargs)

    def load_inputs(self, dtype_override=None, batch_size=1):
        if self.model is None:
            self._load_model_instance()

        image_path = get_file(self.sample_image)
        image = Image.open(str(image_path)).convert("RGB")

        query = f"<image>\n{self.sample_text}"
        prompt, input_ids, pixel_values = self.model.preprocess_inputs(
            query, [image], max_partition=9
        )

        attention_mask = torch.ne(input_ids, self.text_tokenizer.pad_token_id)

        if dtype_override is not None and pixel_values is not None:
            pixel_values = pixel_values.to(dtype_override)

        inputs = {
            "input_ids": input_ids.unsqueeze(0),
            "attention_mask": attention_mask.unsqueeze(0),
        }

        if pixel_values is not None:
            inputs["pixel_values"] = [pixel_values]

        if batch_size > 1:
            for key in inputs:
                if torch.is_tensor(inputs[key]):
                    inputs[key] = inputs[key].repeat_interleave(batch_size, dim=0)

        return inputs

    def decode_output(self, **kwargs):
        outputs = kwargs.get("outputs")
        if outputs is None:
            return None

        if self.text_tokenizer is None and self.model is not None:
            self.text_tokenizer = self.model.get_text_tokenizer()

        if self.text_tokenizer is None:
            return None

        if isinstance(outputs, torch.Tensor):
            if outputs.dtype in (torch.long, torch.int32, torch.int64):
                token_ids = outputs
            else:
                token_ids = outputs.argmax(dim=-1)
        else:
            token_ids = outputs

        return self.text_tokenizer.decode(token_ids[0], skip_special_tokens=True)
