# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
CAFormer model loader implementation
"""

from typing import Optional

import timm

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
from ...tools.utils import (
    VisionPreprocessor,
    VisionPostprocessor,
)
from datasets import load_dataset


class ModelVariant(StrEnum):
    """Available CAFormer model variants."""

    S36_SAIL_IN22K_FT_IN1K_384 = "S36_Sail_In22k_Ft_In1k_384"


class ModelLoader(ForgeModel):
    """CAFormer model loader implementation."""

    _VARIANTS = {
        ModelVariant.S36_SAIL_IN22K_FT_IN1K_384: ModelConfig(
            pretrained_model_name="hf_hub:timm/caformer_s36.sail_in22k_ft_in1k_384",
        ),
    }

    DEFAULT_VARIANT = ModelVariant.S36_SAIL_IN22K_FT_IN1K_384

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
            model="CAFormer",
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

    def input_preprocess(self, dtype_override=None, batch_size=1, image=None):
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

    def load_inputs(self, dtype_override=None, batch_size=1, image=None):
        if image is None:
            dataset = load_dataset("huggingface/cats-image", split="test")
            image = dataset[0]["image"]
        return self.input_preprocess(
            image=image,
            dtype_override=dtype_override,
            batch_size=batch_size,
        )

    def output_postprocess(self, output, top_k=1):
        if self._postprocessor is None:
            model_name = self._variant_config.pretrained_model_name
            self._postprocessor = VisionPostprocessor(
                model_source=ModelSource.TIMM,
                model_name=model_name,
                model_instance=self.model,
                use_1k_labels=True,
            )
        return self._postprocessor.postprocess(output, top_k=top_k, return_dict=True)
