# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
H-optimus-0 model loader implementation for image feature extraction.
"""

from typing import Optional

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


class ModelVariant(StrEnum):
    """Available H-optimus-0 model variants."""

    H_OPTIMUS_0 = "H-optimus-0"


class ModelLoader(ForgeModel):
    """H-optimus-0 model loader implementation for image feature extraction."""

    _VARIANTS = {
        ModelVariant.H_OPTIMUS_0: ModelConfig(
            pretrained_model_name="hf-hub:bioptimus/H-optimus-0",
        ),
    }

    DEFAULT_VARIANT = ModelVariant.H_OPTIMUS_0

    def __init__(self, variant: Optional[ModelVariant] = None):
        """Initialize ModelLoader with specified variant.

        Args:
            variant: Optional ModelVariant specifying which variant to use.
                     If None, DEFAULT_VARIANT is used.
        """
        super().__init__(variant)

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
            model="H-optimus-0",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.CV_IMAGE_FE,
            source=ModelSource.TIMM,
            framework=Framework.TORCH,
        )

    def load_model(self, *, dtype_override=None, **kwargs):
        """Load the H-optimus-0 model.

        Args:
            dtype_override: Optional torch.dtype to override the model's default dtype.

        Returns:
            torch.nn.Module: The H-optimus-0 model instance.
        """
        import timm

        model_name = self._variant_config.pretrained_model_name

        model = timm.create_model(
            model_name,
            pretrained=True,
            init_values=1e-5,
            dynamic_img_size=False,
        )
        model.eval()

        if dtype_override is not None:
            model = model.to(dtype_override)

        return model

    def load_inputs(self, dtype_override=None):
        """Load and return sample inputs for the H-optimus-0 model.

        Args:
            dtype_override: Optional torch.dtype to override the input dtype.

        Returns:
            torch.Tensor: Preprocessed input tensor.
        """
        from torchvision import transforms
        from datasets import load_dataset

        transform = transforms.Compose(
            [
                transforms.Resize(224),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=(0.707223, 0.578729, 0.703617),
                    std=(0.211883, 0.230117, 0.177517),
                ),
            ]
        )

        dataset = load_dataset("huggingface/cats-image", split="test")
        image = dataset[0]["image"]

        pixel_values = transform(image).unsqueeze(0)

        if dtype_override is not None:
            pixel_values = pixel_values.to(dtype_override)

        return pixel_values
