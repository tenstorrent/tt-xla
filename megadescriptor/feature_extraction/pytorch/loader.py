# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
MegaDescriptor model loader implementation for feature extraction (PyTorch).
"""

import timm
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
class MegaDescriptorConfig(ModelConfig):
    """Configuration specific to MegaDescriptor models"""

    source: ModelSource = ModelSource.TIMM


class ModelVariant(StrEnum):
    """Available MegaDescriptor feature extraction model variants."""

    BASE_224 = "Base_224"


class ModelLoader(ForgeModel):
    """MegaDescriptor model loader implementation for feature extraction (PyTorch)."""

    _VARIANTS = {
        ModelVariant.BASE_224: MegaDescriptorConfig(
            pretrained_model_name="hf-hub:BVRA/MegaDescriptor-B-224",
            source=ModelSource.TIMM,
        ),
    }

    DEFAULT_VARIANT = ModelVariant.BASE_224

    def __init__(self, variant: Optional[ModelVariant] = None):
        """Initialize ModelLoader with specified variant.

        Args:
            variant: Optional ModelVariant specifying which variant to use.
                     If None, DEFAULT_VARIANT is used.
        """
        super().__init__(variant)
        self.model = None
        self._preprocessor = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        """Get model information for dashboard and metrics reporting.

        Args:
            variant: Optional ModelVariant specifying which variant to use.
                     If None, DEFAULT_VARIANT is used.

        Returns:
            ModelInfo: Information about the model and variant
        """
        if variant is None:
            variant = cls.DEFAULT_VARIANT

        return ModelInfo(
            model="MegaDescriptor",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.CV_IMAGE_FE,
            source=ModelSource.TIMM,
            framework=Framework.TORCH,
        )

    def load_model(self, *, dtype_override=None, **kwargs):
        """Load and return the MegaDescriptor model instance.

        Args:
            dtype_override: Optional torch.dtype to override the model's default dtype.

        Returns:
            torch.nn.Module: The MegaDescriptor model instance for feature extraction.
        """
        model_name = self._variant_config.pretrained_model_name

        model = timm.create_model(model_name, pretrained=True)
        model.eval()

        self.model = model

        if self._preprocessor is not None:
            self._preprocessor.set_cached_model(model)

        if dtype_override is not None:
            model = model.to(dtype_override)

        return model

    def load_inputs(self, dtype_override=None, batch_size=1, image=None):
        """Load and return sample inputs for the MegaDescriptor model.

        Args:
            dtype_override: Optional torch.dtype override.
            batch_size: Batch size (default: 1).
            image: Optional input image. If None, loads from HuggingFace datasets.

        Returns:
            torch.Tensor: Preprocessed input tensor.
        """
        if image is None:
            dataset = load_dataset("huggingface/cats-image", split="test")
            image = dataset[0]["image"]
        return self.input_preprocess(
            image=image,
            dtype_override=dtype_override,
            batch_size=batch_size,
        )

    def input_preprocess(self, dtype_override=None, batch_size=1, image=None):
        """Preprocess input image(s) and return model-ready input tensor.

        Args:
            dtype_override: Optional torch.dtype override (default: float32).
            batch_size: Batch size (ignored if image is a list).
            image: PIL Image, URL string, tensor, list of images/URLs, or None.

        Returns:
            torch.Tensor: Preprocessed input tensor.
        """
        if self._preprocessor is None:
            model_name = self._variant_config.pretrained_model_name
            source = self._variant_config.source

            self._preprocessor = VisionPreprocessor(
                model_source=source,
                model_name=model_name,
            )

            if self.model is not None:
                self._preprocessor.set_cached_model(self.model)

        model_for_config = None
        if self.model is not None:
            model_for_config = self.model

        return self._preprocessor.preprocess(
            image=image,
            dtype_override=dtype_override,
            batch_size=batch_size,
            model_for_config=model_for_config,
        )
