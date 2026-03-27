# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
SAM ViT model loader implementation for feature extraction (PyTorch/TIMM).
"""

import timm
from datasets import load_dataset
from typing import Optional

from ....config import (
    ModelConfig,
    ModelInfo,
    ModelGroup,
    ModelTask,
    ModelSource,
    Framework,
    StrEnum,
)
from ....base import ForgeModel
from ....tools.utils import VisionPreprocessor


class ModelVariant(StrEnum):
    """Available SAM ViT feature extraction model variants."""

    SAMVIT_BASE_PATCH16_SA1B = "samvit_base_patch16_sa1b"


class ModelLoader(ForgeModel):
    """SAM ViT model loader implementation for feature extraction (PyTorch/TIMM)."""

    _VARIANTS = {
        ModelVariant.SAMVIT_BASE_PATCH16_SA1B: ModelConfig(
            pretrained_model_name="samvit_base_patch16.sa1b",
        ),
    }

    DEFAULT_VARIANT = ModelVariant.SAMVIT_BASE_PATCH16_SA1B

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self.model = None
        self._preprocessor = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT

        return ModelInfo(
            model="SAM ViT",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.CV_IMAGE_FE,
            source=ModelSource.TIMM,
            framework=Framework.TORCH,
        )

    def load_model(self, *, dtype_override=None, **kwargs):
        model_name = self._variant_config.pretrained_model_name

        model = timm.create_model(model_name, pretrained=True)
        model.eval()

        self.model = model

        if self._preprocessor is not None:
            self._preprocessor.set_cached_model(model)

        if dtype_override is not None:
            model = model.to(dtype_override)

        return model

    def input_preprocess(self, dtype_override=None, batch_size=1, image=None):
        if self._preprocessor is None:
            model_name = self._variant_config.pretrained_model_name

            self._preprocessor = VisionPreprocessor(
                model_source=ModelSource.TIMM,
                model_name=model_name,
            )

            if hasattr(self, "model") and self.model is not None:
                self._preprocessor.set_cached_model(self.model)

        model_for_config = None
        if hasattr(self, "model") and self.model is not None:
            model_for_config = self.model

        return self._preprocessor.preprocess(
            image=image,
            dtype_override=dtype_override,
            batch_size=batch_size,
            model_for_config=model_for_config,
        )

    def load_inputs(self, dtype_override=None, batch_size=1, image=None):
        if image is None:
            dataset = load_dataset("huggingface/cats-image", split="test")
            image = dataset[0]["image"]
        return self.input_preprocess(
            image=image,
            dtype_override=dtype_override,
            batch_size=batch_size,
        )
