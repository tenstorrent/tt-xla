# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Hiera model loader implementation
"""

from typing import Optional
from dataclasses import dataclass

from transformers import AutoModelForPreTraining, AutoImageProcessor

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
from datasets import load_dataset


@dataclass
class HieraConfig(ModelConfig):
    """Configuration specific to Hiera models"""

    source: ModelSource


class ModelVariant(StrEnum):
    """Available Hiera model variants."""

    TINY_224_MAE = "Tiny_224_MAE"


class ModelLoader(ForgeModel):
    """Hiera model loader implementation."""

    _VARIANTS = {
        ModelVariant.TINY_224_MAE: HieraConfig(
            pretrained_model_name="facebook/hiera-tiny-224-mae-hf",
            source=ModelSource.HUGGING_FACE,
        ),
    }

    DEFAULT_VARIANT = ModelVariant.TINY_224_MAE

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self._processor = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT

        source = cls._VARIANTS[variant].source

        return ModelInfo(
            model="Hiera",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.CV_IMAGE_FE,
            source=source,
            framework=Framework.TORCH,
        )

    def load_model(self, *, dtype_override=None, **kwargs):
        model_name = self._variant_config.pretrained_model_name

        model = AutoModelForPreTraining.from_pretrained(model_name, **kwargs)
        model.eval()

        if dtype_override is not None:
            model = model.to(dtype_override)

        return model

    def load_inputs(self, dtype_override=None):
        if self._processor is None:
            self._processor = AutoImageProcessor.from_pretrained(
                self._variant_config.pretrained_model_name
            )

        dataset = load_dataset("huggingface/cats-image", split="test")
        image = dataset[0]["image"]

        inputs = self._processor(image, return_tensors="pt")

        return inputs
