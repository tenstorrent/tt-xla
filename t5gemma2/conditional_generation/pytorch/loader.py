# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
T5Gemma2 model loader implementation for multimodal conditional generation.
"""

import torch
from transformers import AutoProcessor, AutoModelForSeq2SeqLM
from typing import Optional
from PIL import Image

from ....base import ForgeModel
from ....config import (
    LLMModelConfig,
    ModelInfo,
    ModelGroup,
    ModelTask,
    ModelSource,
    Framework,
    StrEnum,
)
from ....tools.utils import cast_input_to_type, get_file


class ModelVariant(StrEnum):
    """Available T5Gemma2 model variants for multimodal conditional generation."""

    T5GEMMA2_2_1B_1B = "t5gemma2_2_1b_1b"


class ModelLoader(ForgeModel):
    """T5Gemma2 model loader implementation for multimodal conditional generation."""

    _VARIANTS = {
        ModelVariant.T5GEMMA2_2_1B_1B: LLMModelConfig(
            pretrained_model_name="google/t5gemma-2-1b-1b",
        ),
    }

    DEFAULT_VARIANT = ModelVariant.T5GEMMA2_2_1B_1B

    sample_text = "Describe this image in detail."
    sample_image_url = "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/p-blog/candy.JPG"

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self.processor = None
        self._model = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        return ModelInfo(
            model="T5Gemma2",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.MM_CONDITIONAL_GENERATION,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def _load_processor(self, dtype_override=None):
        kwargs = {}
        if dtype_override is not None:
            kwargs["torch_dtype"] = dtype_override

        self.processor = AutoProcessor.from_pretrained(
            self._variant_config.pretrained_model_name, **kwargs
        )

        return self.processor

    def load_model(self, *, dtype_override=None, **kwargs):
        pretrained_model_name = self._variant_config.pretrained_model_name

        if self.processor is None:
            self._load_processor(dtype_override=dtype_override)

        model_kwargs = {"use_cache": False}
        if dtype_override is not None:
            model_kwargs["torch_dtype"] = dtype_override
        model_kwargs |= kwargs

        model = AutoModelForSeq2SeqLM.from_pretrained(
            pretrained_model_name, **model_kwargs
        )
        model.eval()
        self._model = model
        return model

    def load_inputs(self, dtype_override=None, batch_size=1):
        if self.processor is None:
            self._load_processor(dtype_override=dtype_override)

        image_file = get_file(self.sample_image_url)
        image = Image.open(image_file).convert("RGB")

        inputs = self.processor(
            text=self.sample_text,
            images=image,
            return_tensors="pt",
        )

        # Encoder-decoder models require decoder_input_ids
        decoder_start_token_id = self._model.config.decoder_start_token_id
        inputs["decoder_input_ids"] = torch.full(
            (1, 1), decoder_start_token_id, dtype=torch.long
        )

        for key in inputs:
            if torch.is_tensor(inputs[key]):
                inputs[key] = inputs[key].repeat_interleave(batch_size, dim=0)

        if dtype_override is not None and "pixel_values" in inputs:
            inputs["pixel_values"] = cast_input_to_type(
                inputs["pixel_values"], dtype_override
            )

        return inputs
