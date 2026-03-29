# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
BEiT v2 model loader implementation
"""

from typing import Optional

import timm
from datasets import load_dataset

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
from ...tools.utils import (
    VisionPreprocessor,
    VisionPostprocessor,
)


class ModelVariant(StrEnum):
    """Available BEiT v2 model variants."""

    BASE_PATCH16_224 = "Base_Patch16_224"


class ModelLoader(ForgeModel):
    """BEiT v2 model loader implementation."""

    _VARIANTS = {
        ModelVariant.BASE_PATCH16_224: ModelConfig(
            pretrained_model_name="beitv2_base_patch16_224.in1k_ft_in22k_in1k",
        ),
    }

    DEFAULT_VARIANT = ModelVariant.BASE_PATCH16_224

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self.model = None
        self._preprocessor = None
        self._postprocessor = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT

        return ModelInfo(
            model="BEiT v2",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.CV_IMAGE_CLS,
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

        if self._postprocessor is not None:
            self._postprocessor.set_model_instance(model)

        if dtype_override is not None:
            model = model.to(dtype_override)

        return model

    def load_inputs(self, dtype_override=None, batch_size=1, image=None):
        if image is None:
            dataset = load_dataset("huggingface/cats-image", split="test")
            image = dataset[0]["image"]

        if self._preprocessor is None:
            model_name = self._variant_config.pretrained_model_name

            self._preprocessor = VisionPreprocessor(
                model_source=ModelSource.TIMM,
                model_name=model_name,
            )

            if self.model is not None:
                self._preprocessor.set_cached_model(self.model)

        model_for_config = self.model if self.model is not None else None

        return self._preprocessor.preprocess(
            image=image,
            dtype_override=dtype_override,
            batch_size=batch_size,
            model_for_config=model_for_config,
        )

    def output_postprocess(self, output):
        if self._postprocessor is None:
            model_name = self._variant_config.pretrained_model_name

            self._postprocessor = VisionPostprocessor(
                model_source=ModelSource.TIMM,
                model_name=model_name,
                model_instance=self.model,
            )

        return self._postprocessor.postprocess(output, top_k=1, return_dict=True)
