# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Granite Docling model loader implementation for image-text-to-text document understanding tasks.
"""
import torch
from PIL import Image
from transformers import AutoModelForImageTextToText, AutoProcessor
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
    """Available Granite Docling model variants."""

    GRANITE_DOCLING_258M_MLX = "granite_docling_258m_mlx"


class ModelLoader(ForgeModel):
    """Granite Docling model loader for image-text-to-text document understanding."""

    _VARIANTS = {
        ModelVariant.GRANITE_DOCLING_258M_MLX: ModelConfig(
            pretrained_model_name="ibm-granite/granite-docling-258M-mlx",
        ),
    }

    DEFAULT_VARIANT = ModelVariant.GRANITE_DOCLING_258M_MLX

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self.processor = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        return ModelInfo(
            model="granite_docling",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.MM_IMAGE_TTT,
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

        model_kwargs = {}
        if dtype_override is not None:
            model_kwargs["torch_dtype"] = dtype_override
        model_kwargs |= kwargs

        model = AutoModelForImageTextToText.from_pretrained(
            pretrained_model_name, **model_kwargs
        )
        model.eval()
        return model

    def load_inputs(self, dtype_override=None, batch_size=1):
        if self.processor is None:
            self._load_processor(dtype_override=dtype_override)

        # Create a sample document page image
        image = Image.new("RGB", (512, 512), color=(255, 255, 255))

        prompt = "Convert this page to docling."
        inputs = self.processor(text=prompt, images=image, return_tensors="pt")

        if dtype_override is not None:
            for key in inputs:
                if inputs[key].dtype.is_floating_point:
                    inputs[key] = inputs[key].to(dtype_override)

        if batch_size > 1:
            for key in inputs:
                inputs[key] = inputs[key].repeat_interleave(batch_size, dim=0)

        return dict(inputs)

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

        generated_text = self.processor.batch_decode(
            predicted_ids, skip_special_tokens=True
        )
        return generated_text
