# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
UForm3 model loader implementation for image-text similarity.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

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
from datasets import load_dataset


class ModelVariant(StrEnum):
    """Available UForm3 model variants for image-text similarity."""

    IMAGE_TEXT_MULTILINGUAL_BASE = "Image_Text_Multilingual_Base"


class UForm3ImageTextModel(nn.Module):
    """Wrapper combining UForm3 text and image encoders into a single module."""

    def __init__(self, text_encoder, image_encoder):
        super().__init__()
        self.text_encoder = text_encoder
        self.image_encoder = image_encoder

    def forward(self, images, input_ids, attention_mask):
        image_embeddings = self.image_encoder(images)
        text_embeddings = self.text_encoder(input_ids, attention_mask)
        return image_embeddings, text_embeddings


class ModelLoader(ForgeModel):
    """UForm3 model loader implementation for image-text similarity tasks."""

    _VARIANTS = {
        ModelVariant.IMAGE_TEXT_MULTILINGUAL_BASE: ModelConfig(
            pretrained_model_name="unum-cloud/uform3-image-text-multilingual-base",
        ),
    }

    DEFAULT_VARIANT = ModelVariant.IMAGE_TEXT_MULTILINGUAL_BASE

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self._text_processor = None
        self._image_processor = None
        self.text_prompts = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        return ModelInfo(
            model="UForm3",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.MM_IMAGE_TEXT_SIM,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def _load_uform(self):
        """Load processors and models from the uform package."""
        from uform import get_model, Modality

        processors, models = get_model(
            self._variant_config.pretrained_model_name,
            modalities=[Modality.TEXT_ENCODER, Modality.IMAGE_ENCODER],
        )
        self._text_processor = processors[Modality.TEXT_ENCODER]
        self._image_processor = processors[Modality.IMAGE_ENCODER]
        self._text_encoder = models[Modality.TEXT_ENCODER]
        self._image_encoder = models[Modality.IMAGE_ENCODER]

    def load_model(self, *, dtype_override=None, **kwargs):
        """Load and return the UForm3 model instance.

        Args:
            dtype_override: Optional torch.dtype to override the model's default dtype.

        Returns:
            torch.nn.Module: The UForm3 wrapper model for image-text similarity.
        """
        if not hasattr(self, "_text_encoder"):
            self._load_uform()

        model = UForm3ImageTextModel(
            text_encoder=self._text_encoder,
            image_encoder=self._image_encoder,
        )

        if dtype_override is not None:
            model = model.to(dtype_override)

        model.eval()
        return model

    def load_inputs(self, dtype_override=None, batch_size=1):
        """Load and return sample inputs for the UForm3 model.

        Args:
            dtype_override: Optional torch.dtype to override the input dtype.
            batch_size: Optional batch size (default 1).

        Returns:
            dict: Input tensors containing images, input_ids, and attention_mask.
        """
        if self._text_processor is None or self._image_processor is None:
            self._load_uform()

        # Load image from HuggingFace dataset
        dataset = load_dataset("huggingface/cats-image")["test"]
        image = dataset[0]["image"]

        # Define text prompts for image-text similarity
        self.text_prompts = ["a photo of a cat", "a photo of a dog"]

        # Process image and text
        image_data = self._image_processor(image)
        text_data = self._text_processor(self.text_prompts)

        inputs = {
            "images": image_data["images"],
            "input_ids": text_data["input_ids"],
            "attention_mask": text_data["attention_mask"],
        }

        # Replicate tensors for batch size
        if batch_size > 1:
            for key in inputs:
                inputs[key] = inputs[key].repeat_interleave(batch_size, dim=0)

        # Convert floating point inputs to dtype_override if specified
        if dtype_override is not None:
            for key in inputs:
                if inputs[key].dtype == torch.float32:
                    inputs[key] = inputs[key].to(dtype_override)

        return inputs

    def post_process(self, outputs):
        """Post-process UForm3 outputs to extract similarity scores.

        Args:
            outputs: Raw model output (image_embeddings, text_embeddings)
        """
        if self.text_prompts is None:
            self.text_prompts = ["a photo of a cat", "a photo of a dog"]

        image_embeddings, text_embeddings = outputs
        image_embeddings = F.normalize(image_embeddings, dim=-1)
        text_embeddings = F.normalize(text_embeddings, dim=-1)

        similarity = image_embeddings @ text_embeddings.T

        for i, text in enumerate(self.text_prompts):
            print(f"Similarity to '{text}':", similarity[0, i].item())
