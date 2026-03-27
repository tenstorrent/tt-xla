# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
SigLIP2 Decoder model loader implementation for image-to-image reconstruction.
"""

from typing import Optional
from transformers import ViTMAEForPreTraining, AutoImageProcessor
from datasets import load_dataset

from ...config import (
    ModelConfig,
    ModelInfo,
    ModelGroup,
    ModelTask,
    ModelSource,
    Framework,
    StrEnum,
)
from ...base import ForgeModel


class ModelVariant(StrEnum):
    """Available SigLIP2 Decoder model variants."""

    BASE = "Base"


class ModelLoader(ForgeModel):
    """SigLIP2 Decoder model loader implementation for image-to-image reconstruction."""

    _VARIANTS = {
        ModelVariant.BASE: ModelConfig(
            pretrained_model_name="nyu-visionx/siglip2_decoder",
        ),
    }

    DEFAULT_VARIANT = ModelVariant.BASE

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self.feature_extractor = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT

        return ModelInfo(
            model="SigLIP2_Decoder",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.CV_IMG_TO_IMG,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def load_model(self, *, dtype_override=None, **kwargs):
        model_name = self._variant_config.pretrained_model_name

        self.feature_extractor = AutoImageProcessor.from_pretrained(model_name)

        model_kwargs = {}
        if dtype_override is not None:
            model_kwargs["torch_dtype"] = dtype_override
        model_kwargs |= kwargs

        model = ViTMAEForPreTraining.from_pretrained(model_name, **model_kwargs)
        model.eval()

        return model

    def load_inputs(self, dtype_override=None, batch_size=1):
        if self.feature_extractor is None:
            model_name = self._variant_config.pretrained_model_name
            self.feature_extractor = AutoImageProcessor.from_pretrained(model_name)

        dataset = load_dataset("huggingface/cats-image", split="test")
        image = dataset[0]["image"]

        inputs = self.feature_extractor(images=image, return_tensors="pt").pixel_values

        inputs = inputs.repeat_interleave(batch_size, dim=0)

        if dtype_override is not None:
            inputs = inputs.to(dtype_override)

        return inputs
