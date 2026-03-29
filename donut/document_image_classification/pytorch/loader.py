# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Donut model loader implementation for document image classification tasks.
"""
from datasets import load_dataset
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
    """Available Donut model variants for document image classification."""

    RVLCDIP = "rvlcdip"


class ModelLoader(ForgeModel):
    """Donut model loader implementation for document image classification tasks."""

    _VARIANTS = {
        ModelVariant.RVLCDIP: ModelConfig(
            pretrained_model_name="naver-clova-ix/donut-base-finetuned-rvlcdip",
        ),
    }

    DEFAULT_VARIANT = ModelVariant.RVLCDIP

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self.processor = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        return ModelInfo(
            model="donut",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.CV_IMAGE_CLS,
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

        dataset = load_dataset("hf-internal-testing/example-documents", split="test")
        image = dataset[1]["image"]

        task_prompt = "<s_rvlcdip>"
        decoder_input_ids = self.processor.tokenizer(
            task_prompt, add_special_tokens=False, return_tensors="pt"
        ).input_ids

        pixel_values = self.processor(image, return_tensors="pt").pixel_values

        if dtype_override is not None:
            pixel_values = pixel_values.to(dtype_override)

        if batch_size > 1:
            pixel_values = pixel_values.repeat_interleave(batch_size, dim=0)
            decoder_input_ids = decoder_input_ids.repeat_interleave(batch_size, dim=0)

        return {
            "pixel_values": pixel_values,
            "decoder_input_ids": decoder_input_ids,
        }

    def unpack_forward_output(self, fwd_output):
        if hasattr(fwd_output, "logits"):
            return fwd_output.logits
        return fwd_output
