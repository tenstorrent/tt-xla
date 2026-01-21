# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
SigLIP model loader implementation for image-text similarity.
"""
import torch
from transformers import AutoProcessor, AutoModel
from typing import Optional
from PIL import Image

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
from ....tools.utils import get_file


class ModelVariant(StrEnum):
    """Available SigLIP model variants for image-text similarity."""

    BASE_PATCH16_224 = "base_patch16_224"
    BASE_PATCH16_256 = "base_patch16_256"
    BASE_PATCH16_384 = "base_patch16_384"
    BASE_PATCH16_512 = "base_patch16_512"
    BASE_PATCH16_256_MULTILINGUAL = "base_patch16_256_multilingual"
    LARGE_PATCH16_256 = "large_patch16_256"
    LARGE_PATCH16_384 = "large_patch16_384"
    SO400M_PATCH14_224 = "so400m_patch14_224"
    SO400M_PATCH14_384 = "so400m_patch14_384"
    SO400M_PATCH16_256_I18N = "so400m_patch16_256_i18n"


class ModelLoader(ForgeModel):
    """SigLIP model loader implementation for image-text similarity tasks."""

    # Dictionary of available model variants using structured configs
    _VARIANTS = {
        ModelVariant.BASE_PATCH16_224: ModelConfig(
            pretrained_model_name="google/siglip-base-patch16-224",
        ),
        ModelVariant.BASE_PATCH16_256: ModelConfig(
            pretrained_model_name="google/siglip-base-patch16-256",
        ),
        ModelVariant.BASE_PATCH16_384: ModelConfig(
            pretrained_model_name="google/siglip-base-patch16-384",
        ),
        ModelVariant.BASE_PATCH16_512: ModelConfig(
            pretrained_model_name="google/siglip-base-patch16-512",
        ),
        ModelVariant.BASE_PATCH16_256_MULTILINGUAL: ModelConfig(
            pretrained_model_name="google/siglip-base-patch16-256-multilingual",
        ),
        ModelVariant.LARGE_PATCH16_256: ModelConfig(
            pretrained_model_name="google/siglip-large-patch16-256",
        ),
        ModelVariant.LARGE_PATCH16_384: ModelConfig(
            pretrained_model_name="google/siglip-large-patch16-384",
        ),
        ModelVariant.SO400M_PATCH14_224: ModelConfig(
            pretrained_model_name="google/siglip-so400m-patch14-224",
        ),
        ModelVariant.SO400M_PATCH14_384: ModelConfig(
            pretrained_model_name="google/siglip-so400m-patch14-384",
        ),
        ModelVariant.SO400M_PATCH16_256_I18N: ModelConfig(
            pretrained_model_name="google/siglip-so400m-patch16-256-i18n",
        ),
    }

    # Default variant to use
    DEFAULT_VARIANT = ModelVariant.BASE_PATCH16_224

    def __init__(self, variant: Optional[ModelVariant] = None):
        """Initialize ModelLoader with specified variant.

        Args:
            variant: Optional ModelVariant specifying which variant to use.
                     If None, DEFAULT_VARIANT is used.
        """
        super().__init__(variant)

        # Configuration parameters
        self.processor = None
        self.text_prompts = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        """Implementation method for getting model info with validated variant.

        Args:
            variant: Optional ModelVariant specifying which variant to use.
                     If None, DEFAULT_VARIANT is used.

        Returns:
            ModelInfo: Information about the model and variant
        """
        return ModelInfo(
            model="siglip",
            variant=variant,
            group=(
                ModelGroup.RED
                if variant == ModelVariant.BASE_PATCH16_224
                else ModelGroup.GENERALITY
            ),
            task=ModelTask.MM_IMAGE_TEXT_SIM,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def _load_processor(self):
        """Load processor for the current variant.

        Returns:
            The loaded processor instance
        """
        # Load the processor
        self.processor = AutoProcessor.from_pretrained(
            self._variant_config.pretrained_model_name
        )

        return self.processor

    def load_model(self, dtype_override=None):
        """Load and return the SigLIP model instance for this instance's variant.

        Args:
            dtype_override: Optional torch.dtype to override the model's default dtype.
                           If not provided, the model will use its default dtype (typically float32).

        Returns:
            torch.nn.Module: The SigLIP model instance for image-text similarity.
        """
        # Get the pretrained model name from the instance's variant config
        pretrained_model_name = self._variant_config.pretrained_model_name

        model_kwargs = {"return_dict": False}

        # Load the model with dtype override if specified
        if dtype_override is not None:
            model_kwargs["torch_dtype"] = dtype_override

        model = AutoModel.from_pretrained(pretrained_model_name, **model_kwargs)
        model.eval()

        return model

    def load_inputs(self, dtype_override=None, batch_size=1):
        """Load and return sample inputs for the SigLIP model with this instance's variant settings.

        Args:
            dtype_override: Optional torch.dtype to override the model's default dtype.
                           If not provided, the model will use its default dtype (typically float32).
            batch_size: Optional batch size to override the default batch size of 1.

        Returns:
            dict: Input tensors that can be fed to the model.
        """
        # Ensure processor is initialized
        if self.processor is None:
            self._load_processor()

        # Get the Image
        image_file = get_file("http://images.cocodataset.org/val2017/000000039769.jpg")
        image = Image.open(image_file)

        # Define text prompts for image-text similarity
        self.text_prompts = ["a photo of 2 cats", "a photo of 2 dogs"]

        # Process both text and images
        inputs = self.processor(
            text=self.text_prompts,
            images=image,
            return_tensors="pt",
            padding="max_length",
        )

        # Replicate tensors for batch size
        for key in inputs:
            if torch.is_tensor(inputs[key]):
                inputs[key] = inputs[key].repeat_interleave(batch_size, dim=0)

        # Convert the input dtype to dtype_override if specified
        if dtype_override is not None:
            for key in inputs:
                if torch.is_tensor(inputs[key]) and inputs[key].dtype == torch.float32:
                    inputs[key] = inputs[key].to(dtype_override)

        return inputs
