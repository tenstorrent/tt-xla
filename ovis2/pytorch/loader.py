# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Ovis2 model loader implementation for multimodal conditional generation.
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
from ...tools.utils import get_file, cast_input_to_type


class ModelVariant(StrEnum):
    """Available Ovis2 model variants."""

    OVIS2_1B = "1B"


class ModelLoader(ForgeModel):
    """Ovis2 model loader for multimodal conditional generation."""

    _VARIANTS = {
        ModelVariant.OVIS2_1B: ModelConfig(
            pretrained_model_name="AIDC-AI/Ovis2-1B",
        ),
    }

    DEFAULT_VARIANT = ModelVariant.OVIS2_1B

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self.text_tokenizer = None
        self.visual_tokenizer = None
        self.model = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT

        return ModelInfo(
            model="Ovis2",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.MM_CONDITIONAL_GENERATION,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def load_model(self, *, dtype_override=None, **kwargs):
        pretrained_model_name = self._variant_config.pretrained_model_name

        model_kwargs = {
            "trust_remote_code": True,
            "multimodal_max_length": 8192,
        }
        if dtype_override is not None:
            model_kwargs["torch_dtype"] = dtype_override
        model_kwargs |= kwargs

        model = AutoModelForCausalLM.from_pretrained(
            pretrained_model_name, **model_kwargs
        )
        model.eval()
        self.model = model

        self.text_tokenizer = model.get_text_tokenizer()
        self.visual_tokenizer = model.get_visual_tokenizer()

        return model

    def load_inputs(self, dtype_override=None, batch_size=1):
        if self.model is None:
            self.load_model(dtype_override=dtype_override)

        image_file = get_file(
            "https://cdn.britannica.com/61/93061-050-99147DCE/Statue-of-Liberty-Island-New-York-Bay.jpg"
        )
        image = Image.open(image_file)

        query = "<image>\nWhat is shown in this image?"

        prompt, input_ids, pixel_values = self.model.preprocess_inputs(
            query, [image], max_partition=9
        )

        attention_mask = torch.ne(input_ids, self.text_tokenizer.pad_token_id)
        input_ids = input_ids.unsqueeze(0)
        attention_mask = attention_mask.unsqueeze(0)

        if pixel_values is not None:
            pixel_values = [pixel_values.to(dtype=self.visual_tokenizer.dtype)]

        if dtype_override is not None:
            input_ids = cast_input_to_type(input_ids, dtype_override)
            attention_mask = cast_input_to_type(attention_mask, dtype_override)

        if batch_size > 1:
            input_ids = input_ids.repeat_interleave(batch_size, dim=0)
            attention_mask = attention_mask.repeat_interleave(batch_size, dim=0)
            if pixel_values is not None:
                pixel_values = pixel_values * batch_size

        inputs = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
        }
        if pixel_values is not None:
            inputs["pixel_values"] = pixel_values

        return inputs

    def decode_output(self, outputs, input_length=None):
        if isinstance(outputs, str):
            return outputs

        if self.text_tokenizer is None and self.model is not None:
            self.text_tokenizer = self.model.get_text_tokenizer()

        if torch.is_tensor(outputs) and outputs.dtype in [torch.long, torch.int]:
            if input_length is not None:
                outputs = outputs[:, input_length:]
            return self.text_tokenizer.batch_decode(outputs, skip_special_tokens=True)[
                0
            ]
        else:
            logits = outputs.logits if hasattr(outputs, "logits") else outputs[0]
            next_token_id = torch.argmax(logits[:, -1, :], dim=-1)
            return self.text_tokenizer.decode(next_token_id)
