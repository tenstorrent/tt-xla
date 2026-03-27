# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
ColQwen2 model loader implementation for visual document retrieval.
"""
from PIL import Image
from transformers import ColQwen2ForRetrieval, AutoProcessor
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


class ModelVariant(StrEnum):
    """Available ColQwen2 model variants for visual document retrieval."""

    MICHAELFEIL_COLQWEN2_V0_1 = "michaelfeil/colqwen2-v0.1"


class ModelLoader(ForgeModel):
    """ColQwen2 model loader implementation for visual document retrieval."""

    _VARIANTS = {
        ModelVariant.MICHAELFEIL_COLQWEN2_V0_1: ModelConfig(
            pretrained_model_name="michaelfeil/colqwen2-v0.1",
        ),
    }

    DEFAULT_VARIANT = ModelVariant.MICHAELFEIL_COLQWEN2_V0_1

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self.processor = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        return ModelInfo(
            model="ColQwen2",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.MM_IMAGE_TEXT_SIM,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def _load_processor(self):
        self.processor = AutoProcessor.from_pretrained(
            self._variant_config.pretrained_model_name,
            trust_remote_code=True,
        )
        return self.processor

    def load_model(self, *, dtype_override=None, **kwargs):
        pretrained_model_name = self._variant_config.pretrained_model_name

        if self.processor is None:
            self._load_processor()

        model_kwargs = {}
        if dtype_override is not None:
            model_kwargs["torch_dtype"] = dtype_override
        model_kwargs |= kwargs

        model = ColQwen2ForRetrieval.from_pretrained(
            pretrained_model_name,
            trust_remote_code=True,
            **model_kwargs,
        )
        model.eval()

        return model

    def load_inputs(self, dtype_override=None):
        if self.processor is None:
            self._load_processor()

        # ColQwen2 processes images for document retrieval
        image = Image.new("RGB", (224, 224), color=(255, 255, 255))
        inputs = self.processor(images=[image], return_tensors="pt")

        if dtype_override is not None and "pixel_values" in inputs:
            inputs["pixel_values"] = inputs["pixel_values"].to(dtype_override)

        return inputs

    def decode_output(self, outputs, inputs=None):
        if hasattr(outputs, "embeddings"):
            return outputs.embeddings
        elif isinstance(outputs, (tuple, list)):
            return outputs[0]
        return outputs

    def unpack_forward_output(self, fwd_output):
        if hasattr(fwd_output, "embeddings") and fwd_output.embeddings is not None:
            return fwd_output.embeddings.flatten()
        return fwd_output
