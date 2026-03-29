# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
UniChart model loader implementation for chart understanding tasks.
"""
import torch
from PIL import Image
from transformers import DonutProcessor, VisionEncoderDecoderModel
from typing import Optional

from ....base import ForgeModel
from ....config import (
    ModelConfig,
    ModelInfo,
    ModelGroup,
    ModelTask,
    ModelSource,
    Framework,
    StrEnum,
)


class ModelVariant(StrEnum):
    """Available UniChart model variants for chart understanding tasks."""

    BASE_960 = "base_960"


class ModelLoader(ForgeModel):
    """UniChart model loader implementation for chart understanding tasks."""

    _VARIANTS = {
        ModelVariant.BASE_960: ModelConfig(
            pretrained_model_name="ahmed-masry/unichart-base-960",
        ),
    }

    DEFAULT_VARIANT = ModelVariant.BASE_960

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self.processor = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        return ModelInfo(
            model="unichart",
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

        self.processor = DonutProcessor.from_pretrained(
            self._variant_config.pretrained_model_name, **kwargs
        )

        return self.processor

    def load_model(self, *, dtype_override=None, **kwargs):
        pretrained_model_name = self._variant_config.pretrained_model_name

        if self.processor is None:
            self._load_processor(dtype_override=dtype_override)

        model_kwargs = {}
        if dtype_override is not None:
            model_kwargs["torch_dtype"] = dtype_override
        model_kwargs |= kwargs

        model = VisionEncoderDecoderModel.from_pretrained(
            pretrained_model_name, **model_kwargs
        )
        model.eval()
        return model

    def load_inputs(self, dtype_override=None, batch_size=1):
        if self.processor is None:
            self._load_processor(dtype_override=dtype_override)

        image = Image.new("RGB", (960, 960), color=(255, 255, 255))
        pixel_values = self.processor(image, return_tensors="pt").pixel_values

        if dtype_override is not None:
            pixel_values = pixel_values.to(dtype_override)

        if batch_size > 1:
            pixel_values = pixel_values.repeat_interleave(batch_size, dim=0)

        input_prompt = "<summarize_chart> <s_answer>"
        decoder_input_ids = self.processor.tokenizer(
            input_prompt, add_special_tokens=False, return_tensors="pt"
        ).input_ids

        if batch_size > 1:
            decoder_input_ids = decoder_input_ids.repeat_interleave(batch_size, dim=0)

        return {"pixel_values": pixel_values, "decoder_input_ids": decoder_input_ids}

    def unpack_forward_output(self, fwd_output):
        if hasattr(fwd_output, "logits"):
            return fwd_output.logits
        return fwd_output

    def decode_output(self, outputs):
        if self.processor is None:
            self._load_processor()

        if hasattr(outputs, "logits"):
            predicted_ids = outputs.logits.argmax(-1)
        else:
            predicted_ids = outputs[0].argmax(-1)

        sequences = self.processor.batch_decode(predicted_ids, skip_special_tokens=True)
        return sequences
