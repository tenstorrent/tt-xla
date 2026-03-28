# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
ViT feature extraction model loader implementation for PyTorch (TIMM).
"""

import timm
from typing import Optional
from datasets import load_dataset

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


class ModelVariant(StrEnum):
    """Available ViT feature extraction model variants."""

    SO400M_PATCH14_SIGLIP_384 = "So400m_Patch14_SigLIP_384"


class ModelLoader(ForgeModel):
    """ViT feature extraction model loader implementation for PyTorch (TIMM)."""

    _VARIANTS = {
        ModelVariant.SO400M_PATCH14_SIGLIP_384: ModelConfig(
            pretrained_model_name="vit_so400m_patch14_siglip_384.webli",
        ),
    }

    DEFAULT_VARIANT = ModelVariant.SO400M_PATCH14_SIGLIP_384

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
        """Implementation method for getting model info with validated variant.

        Args:
            variant: Optional ModelVariant specifying which variant to use.
                     If None, DEFAULT_VARIANT is used.

        Returns:
            ModelInfo: Information about the model and variant
        """
        if variant is None:
            variant = cls.DEFAULT_VARIANT

        return ModelInfo(
            model="ViT",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.CV_IMAGE_FE,
            source=ModelSource.TIMM,
            framework=Framework.TORCH,
        )

    def load_model(self, *, dtype_override=None, **kwargs):
        """Load and return the ViT feature extraction model.

        Args:
            dtype_override: Optional torch.dtype to override the model's default dtype.

        Returns:
            torch.nn.Module: The ViT model instance.
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
        """Load and return sample inputs for the model.

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

        if self._preprocessor is None:
            model_name = self._variant_config.pretrained_model_name
            self._preprocessor = VisionPreprocessor(
                model_source=ModelSource.TIMM,
                model_name=model_name,
            )
            if self.model is not None:
                self._preprocessor.set_cached_model(self.model)

        return self._preprocessor.preprocess(
            image=image,
            dtype_override=dtype_override,
            batch_size=batch_size,
            model_for_config=self.model,
        )
