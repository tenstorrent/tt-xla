# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
DINOv3 ViT model loader implementation for feature extraction (PyTorch).
"""

import torch
from datasets import load_dataset
from typing import Optional
from dataclasses import dataclass

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
from ....tools.utils import VisionPreprocessor


@dataclass
class DINOv3Config(ModelConfig):
    """Configuration specific to DINOv3 models."""

    source: ModelSource = ModelSource.HUGGING_FACE


class ModelVariant(StrEnum):
    """Available DINOv3 ViT feature extraction model variants."""

    BASE = "Base"
    SMALL_PLUS = "Small+"


class ModelLoader(ForgeModel):
    """DINOv3 ViT model loader implementation for feature extraction (PyTorch)."""

    _VARIANTS = {
        ModelVariant.BASE: DINOv3Config(
            pretrained_model_name="facebook/dinov3-vitb16-pretrain-lvd1689m",
            source=ModelSource.HUGGING_FACE,
        ),
        ModelVariant.SMALL_QKVB: DINOv3Config(
            pretrained_model_name="vit_small_patch16_dinov3_qkvb.lvd1689m",
            source=ModelSource.TIMM,
        ),
        ModelVariant.LARGE: ModelConfig(
            pretrained_model_name="camenduru/dinov3-vitl16-pretrain-lvd1689m",
        ),
        ModelVariant.SMALL_PLUS: ModelConfig(
            pretrained_model_name="facebook/dinov3-vits16plus-pretrain-lvd1689m",
        ),
    }

    DEFAULT_VARIANT = ModelVariant.BASE

    def __init__(self, variant: Optional[ModelVariant] = None):
        """Initialize ModelLoader with specified variant.

        Args:
            variant: Optional ModelVariant specifying which variant to use.
                     If None, DEFAULT_VARIANT is used.
        """
        super().__init__(variant)
        self.processor = None
        self.model = None
        self._preprocessor = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        """Implementation method for getting model info with validated variant.

        Args:
            variant: Optional ModelVariant specifying which variant to use.
                     If None, DEFAULT_VARIANT is used.

        Returns:
            ModelInfo: Information about the model and variant
        """
        if variant is None:
            variant = cls.DEFAULT_VARIANT

        source = cls._VARIANTS[variant].source

        return ModelInfo(
            model="DINOv3 ViT",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.CV_IMAGE_FE,
            source=source,
            framework=Framework.TORCH,
        )

    def _load_processor(self):
        """Load image processor for the current variant (HuggingFace only).

        Returns:
            The loaded processor instance
        """
        from transformers import DINOv3ViTImageProcessor

        pretrained_model_name = self._variant_config.pretrained_model_name
        self.processor = DINOv3ViTImageProcessor.from_pretrained(pretrained_model_name)
        return self.processor

    def load_model(self, *, dtype_override=None, **kwargs):
        """Load and return the DINOv3 ViT model instance.

        Args:
            dtype_override: Optional torch.dtype to override the model's default dtype.

        Returns:
            torch.nn.Module: The DINOv3 ViT model instance for feature extraction.
        """
        pretrained_model_name = self._variant_config.pretrained_model_name
        source = self._variant_config.source

        if source == ModelSource.TIMM:
            import timm

            model = timm.create_model(
                pretrained_model_name, pretrained=True, num_classes=0
            )
            if dtype_override is not None:
                model = model.to(dtype_override)
        elif source == ModelSource.HUGGING_FACE:
            from transformers import DINOv3ViTModel

            model_kwargs = {}
            if dtype_override is not None:
                model_kwargs["torch_dtype"] = dtype_override
            model_kwargs |= kwargs

            model = DINOv3ViTModel.from_pretrained(
                pretrained_model_name, **model_kwargs
            )

        model.eval()
        self.model = model

        if self._preprocessor is not None:
            self._preprocessor.set_cached_model(model)

        return model

    def load_inputs(self, dtype_override=None, batch_size=1, image=None):
        """Load and return sample inputs for the DINOv3 ViT model.

        Args:
            dtype_override: Optional torch.dtype to override the model inputs' default dtype.
            batch_size: Batch size for the inputs.
            image: Optional input image. If None, loads from HuggingFace datasets.

        Returns:
            Model inputs (dict for HuggingFace, tensor for TIMM).
        """
        if image is None:
            dataset = load_dataset("huggingface/cats-image", split="test")
            image = dataset[0]["image"]

        source = self._variant_config.source

        if source == ModelSource.TIMM:
            if self._preprocessor is None:
                model_name = self._variant_config.pretrained_model_name
                self._preprocessor = VisionPreprocessor(
                    model_source=source,
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
        else:
            if self.processor is None:
                self._load_processor()

            inputs = self.processor(images=image, return_tensors="pt")

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
