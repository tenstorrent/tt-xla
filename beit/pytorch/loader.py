# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
BEiT model loader implementation for image classification
"""
import timm
import torch
from transformers import BeitImageProcessor, BeitForImageClassification
from datasets import load_dataset
from typing import Optional
from dataclasses import dataclass

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
from ...tools.utils import VisionPreprocessor, VisionPostprocessor


@dataclass
class BeitConfig(ModelConfig):
    """Configuration specific to BEiT models"""

    source: ModelSource


class ModelVariant(StrEnum):
    """Available BEiT model variants."""

    # HuggingFace variants
    BASE = "Base"
    LARGE = "Large"

    # TIMM variants
    BEIT_BASE_PATCH16_224_IN22K_FT_IN22K_IN1K = (
        "BEiT_Base_Patch16_224_IN22K_FT_IN22K_IN1K"
    )


class ModelLoader(ForgeModel):
    """BEiT model loader implementation for image classification tasks."""

    # Dictionary of available model variants
    _VARIANTS = {
        # HuggingFace variants
        ModelVariant.BASE: BeitConfig(
            pretrained_model_name="microsoft/beit-base-patch16-224",
            source=ModelSource.HUGGING_FACE,
        ),
        ModelVariant.LARGE: BeitConfig(
            pretrained_model_name="microsoft/beit-large-patch16-224",
            source=ModelSource.HUGGING_FACE,
        ),
        # TIMM variants
        ModelVariant.BEIT_BASE_PATCH16_224_IN22K_FT_IN22K_IN1K: BeitConfig(
            pretrained_model_name="beit_base_patch16_224.in22k_ft_in22k_in1k",
            source=ModelSource.TIMM,
        ),
    }

    # Default variant to use
    DEFAULT_VARIANT = ModelVariant.BASE

    def __init__(self, variant: Optional[ModelVariant] = None):
        """Initialize ModelLoader with specified variant.

        Args:
            variant: Optional ModelVariant specifying which variant to use.
                     If None, DEFAULT_VARIANT is used.
        """
        super().__init__(variant)
        self.model = None
        self.processor = None
        self._preprocessor = None
        self._postprocessor = None

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

        if source == ModelSource.TIMM:
            group = ModelGroup.VULCAN
        else:
            group = ModelGroup.GENERALITY

        return ModelInfo(
            model="BEiT",
            variant=variant,
            group=group,
            task=ModelTask.CV_IMAGE_CLS,
            source=source,
            framework=Framework.TORCH,
        )

    def _load_processor(self):
        """Load processor for the current variant.

        Returns:
            The loaded processor instance
        """

        # Initialize processor
        self.processor = BeitImageProcessor.from_pretrained(
            self._variant_config.pretrained_model_name
        )

        return self.processor

    def load_model(self, *, dtype_override=None, **kwargs):
        """Load and return the BEiT model instance for this instance's variant.

        Args:
            dtype_override: Optional torch.dtype to override the model's default dtype.
                           If not provided, the model will use its default dtype (typically float32).

        Returns:
            torch.nn.Module: The BEiT model instance for image classification.
        """
        model_name = self._variant_config.pretrained_model_name
        source = self._variant_config.source

        if source == ModelSource.TIMM:
            model = timm.create_model(model_name, pretrained=True)
        else:
            # Ensure processor is loaded for HuggingFace models
            if self.processor is None:
                self._load_processor()

            model = BeitForImageClassification.from_pretrained(model_name, **kwargs)

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
        """Preprocess input image(s) and return model-ready input tensor.

        Args:
            dtype_override: Optional torch.dtype override (default: float32).
            batch_size: Batch size (ignored if image is a list).
            image: PIL Image, URL string, tensor, list of images/URLs, or None (uses default image).

        Returns:
            torch.Tensor: Preprocessed input tensor.
        """
        source = self._variant_config.source

        if source == ModelSource.TIMM:
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
        else:
            # HuggingFace preprocessing
            if self.processor is None:
                self._load_processor()

            inputs = self.processor(images=image, return_tensors="pt")

            if dtype_override is not None:
                inputs["pixel_values"] = inputs["pixel_values"].to(dtype_override)

            for key in inputs:
                if torch.is_tensor(inputs[key]):
                    inputs[key] = inputs[key].repeat_interleave(batch_size, dim=0)

            return inputs

    def load_inputs(self, dtype_override=None, batch_size=1, image=None):
        """Load and return sample inputs for the BEiT model with this instance's variant settings.

        Args:
            dtype_override: Optional torch.dtype to override the model inputs' default dtype.
            batch_size: Optional batch size to override the default batch size of 1.
            image: Optional input image. If None, loads from HuggingFace datasets.

        Returns:
            Input tensors that can be fed to the model.
        """
        if image is None:
            dataset = load_dataset("huggingface/cats-image", split="test")
            image = dataset[0]["image"]

        return self.input_preprocess(
            image=image,
            dtype_override=dtype_override,
            batch_size=batch_size,
        )

    def output_postprocess(self, output, top_k=1):
        """Post-process model outputs.

        Args:
            output: Model output tensor.
            top_k: Number of top predictions to return (default: 1).

        Returns:
            dict: Prediction dict with top predictions.
        """
        source = self._variant_config.source

        if source == ModelSource.TIMM:
            if self._postprocessor is None:
                model_name = self._variant_config.pretrained_model_name
                self._postprocessor = VisionPostprocessor(
                    model_source=source,
                    model_name=model_name,
                    model_instance=self.model,
                )

            return self._postprocessor.postprocess(
                output, top_k=top_k, return_dict=True
            )
        else:
            # HuggingFace postprocessing
            if hasattr(output, "logits"):
                logits = output.logits
            else:
                logits = output
            predicted_class_idx = logits.argmax(-1).item()
            if self.model is not None and hasattr(self.model.config, "id2label"):
                return {"label": self.model.config.id2label[predicted_class_idx]}
            return {"class_idx": predicted_class_idx}
