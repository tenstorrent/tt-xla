# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
EVA02 model loader implementation
"""

from typing import Optional
from dataclasses import dataclass
import timm
import torch

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


@dataclass
class EVA02Config(ModelConfig):
    """Configuration specific to EVA02 models"""

    source: ModelSource


class ModelVariant(StrEnum):
    """Available EVA02 model variants."""

    ENORMOUS_PATCH14_PLUS_CLIP_224 = "Enormous_Patch14_Plus_CLIP_224"


class ModelLoader(ForgeModel):
    """EVA02 model loader implementation."""

    _VARIANTS = {
        ModelVariant.ENORMOUS_PATCH14_PLUS_CLIP_224: EVA02Config(
            pretrained_model_name="hf_hub:timm/eva02_enormous_patch14_plus_clip_224.laion2b_s9b_b144k",
            source=ModelSource.TIMM,
        ),
    }

    DEFAULT_VARIANT = ModelVariant.ENORMOUS_PATCH14_PLUS_CLIP_224

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self.model = None
        self._preprocessor = None
        self._postprocessor = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT

        source = cls._VARIANTS[variant].source

        return ModelInfo(
            model="EVA02",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.CV_IMAGE_CLS,
            source=source,
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
            source = self._variant_config.source

            self._preprocessor = VisionPreprocessor(
                model_source=source,
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

    def output_postprocess(self, output):
        if self._postprocessor is None:
            model_name = self._variant_config.pretrained_model_name
            source = self._variant_config.source

            self._postprocessor = VisionPostprocessor(
                model_source=source,
                model_name=model_name,
                model_instance=self.model,
            )

        return self._postprocessor.postprocess(output, top_k=1, return_dict=True)
