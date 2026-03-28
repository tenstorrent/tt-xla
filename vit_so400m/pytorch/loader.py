# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
ViT-SO400M (SigLIP2) model loader implementation
"""

from typing import Optional

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
from ...tools.utils import VisionPreprocessor, VisionPostprocessor


class ModelVariant(StrEnum):
    """Available ViT-SO400M model variants."""

    VIT_SO400M_PATCH16_SIGLIP_512 = "So400m_Patch16_SigLIP_512"


class ModelLoader(ForgeModel):
    """ViT-SO400M model loader implementation."""

    _VARIANTS = {
        ModelVariant.VIT_SO400M_PATCH16_SIGLIP_512: ModelConfig(
            pretrained_model_name="vit_so400m_patch16_siglip_512.v2_webli",
        ),
    }

    DEFAULT_VARIANT = ModelVariant.VIT_SO400M_PATCH16_SIGLIP_512

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT

        return ModelInfo(
            model="ViT-SO400M",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.CV_IMAGE_CLS,
            source=ModelSource.TIMM,
            framework=Framework.TORCH,
        )

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self.model = None
        self._preprocessor = None
        self._postprocessor = None

    def load_model(self, *, dtype_override=None, **kwargs):
        import timm

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
        from datasets import load_dataset

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
            )

        return self._postprocessor.postprocess(output, top_k=top_k, return_dict=True)
