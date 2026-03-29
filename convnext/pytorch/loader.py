# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
ConvNeXt model loader implementation
"""

from typing import Optional
from dataclasses import dataclass
import timm
import torch
from transformers import AutoImageProcessor, AutoModelForImageClassification

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
class ConvNeXtConfig(ModelConfig):
    """Configuration specific to ConvNeXt models"""

    source: ModelSource


class ModelVariant(StrEnum):
    """Available ConvNeXt model variants."""

    BASE_CLIP_LAION2B = "Base_CLIP_LAION2B"
    BASE_224 = "Base_224"


class ModelLoader(ForgeModel):
    """ConvNeXt model loader implementation."""

    _VARIANTS = {
        ModelVariant.BASE_CLIP_LAION2B: ConvNeXtConfig(
            pretrained_model_name="hf_hub:timm/convnext_base.clip_laion2b",
            source=ModelSource.TIMM,
        ),
        ModelVariant.BASE_224: ConvNeXtConfig(
            pretrained_model_name="facebook/convnext-base-224",
            source=ModelSource.HUGGING_FACE,
        ),
    }

    DEFAULT_VARIANT = ModelVariant.BASE_CLIP_LAION2B

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self.model = None
        self._processor = None
        self._preprocessor = None
        self._postprocessor = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT

        source = cls._VARIANTS[variant].source

        return ModelInfo(
            model="ConvNeXt",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.CV_IMAGE_CLS,
            source=source,
            framework=Framework.TORCH,
        )

    def load_model(self, *, dtype_override=None, **kwargs):
        model_name = self._variant_config.pretrained_model_name
        source = self._variant_config.source

        if source == ModelSource.HUGGING_FACE:
            model_kwargs = {}
            if dtype_override is not None:
                model_kwargs["torch_dtype"] = dtype_override
            model_kwargs |= kwargs

            model = AutoModelForImageClassification.from_pretrained(
                model_name, **model_kwargs
            )
            model.eval()
        else:
            model = timm.create_model(model_name, pretrained=True)
            model.eval()

            if dtype_override is not None:
                model = model.to(dtype_override)

        self.model = model

        if self._preprocessor is not None:
            self._preprocessor.set_cached_model(model)

        if self._postprocessor is not None:
            self._postprocessor.set_model_instance(model)

        return model

    def load_inputs(self, dtype_override=None, batch_size=1, image=None):
        if image is None:
            dataset = load_dataset("huggingface/cats-image", split="test")
            image = dataset[0]["image"]

        source = self._variant_config.source

        if source == ModelSource.HUGGING_FACE:
            if self._processor is None:
                model_name = self._variant_config.pretrained_model_name
                self._processor = AutoImageProcessor.from_pretrained(model_name)

            inputs = self._processor(images=image, return_tensors="pt")

            for key in inputs:
                if torch.is_tensor(inputs[key]):
                    inputs[key] = inputs[key].repeat_interleave(batch_size, dim=0)

            if dtype_override is not None:
                for key in inputs:
                    if (
                        torch.is_tensor(inputs[key])
                        and inputs[key].dtype.is_floating_point
                    ):
                        inputs[key] = inputs[key].to(dtype_override)

            return inputs
        else:
            if self._preprocessor is None:
                model_name = self._variant_config.pretrained_model_name

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
