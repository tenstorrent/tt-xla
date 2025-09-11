# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
ViT model loader implementation
"""

from transformers import AutoImageProcessor, ViTForImageClassification
from torchvision import models
from PIL import Image
from typing import Optional
from dataclasses import dataclass

from ...config import (
    ModelInfo,
    ModelGroup,
    ModelTask,
    ModelSource,
    ModelConfig,
    Framework,
    StrEnum,
)
from ...base import ForgeModel
from ...tools.utils import get_file, print_compiled_model_results


@dataclass
class ViTConfig(ModelConfig):
    """Configuration specific to ViT models"""

    source: ModelSource


class ModelVariant(StrEnum):
    """Available ViT model variants."""

    # HuggingFace variants
    BASE = "base"
    LARGE = "large"

    # Torchvision variants
    VIT_B_16 = "vit_b_16"
    VIT_B_32 = "vit_b_32"
    VIT_L_16 = "vit_l_16"
    VIT_L_32 = "vit_l_32"
    VIT_H_14 = "vit_h_14"


class ModelLoader(ForgeModel):
    """ViT model loader implementation."""

    # Dictionary of available model variants using structured configs
    _VARIANTS = {
        # HuggingFace variants
        ModelVariant.BASE: ViTConfig(
            pretrained_model_name="google/vit-base-patch16-224",
            source=ModelSource.HUGGING_FACE,
        ),
        ModelVariant.LARGE: ViTConfig(
            pretrained_model_name="google/vit-large-patch16-224",
            source=ModelSource.HUGGING_FACE,
        ),
        # Torchvision variants
        ModelVariant.VIT_B_16: ViTConfig(
            pretrained_model_name="vit_b_16",
            source=ModelSource.TORCHVISION,
        ),
        ModelVariant.VIT_B_32: ViTConfig(
            pretrained_model_name="vit_b_32",
            source=ModelSource.TORCHVISION,
        ),
        ModelVariant.VIT_L_16: ViTConfig(
            pretrained_model_name="vit_l_16",
            source=ModelSource.TORCHVISION,
        ),
        ModelVariant.VIT_L_32: ViTConfig(
            pretrained_model_name="vit_l_32",
            source=ModelSource.TORCHVISION,
        ),
        ModelVariant.VIT_H_14: ViTConfig(
            pretrained_model_name="vit_h_14",
            source=ModelSource.TORCHVISION,
        ),
    }

    # Default variant to use
    DEFAULT_VARIANT = ModelVariant.LARGE

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

        # Get source from variant config
        source = cls._VARIANTS[variant].source

        return ModelInfo(
            model="vit",
            variant=variant,
            group=ModelGroup.RED
            if variant == ModelVariant.BASE
            else ModelGroup.GENERALITY,
            task=ModelTask.CV_IMAGE_CLS,
            source=source,
            framework=Framework.TORCH,
        )

    def __init__(self, variant: Optional[ModelVariant] = None):
        """Initialize ModelLoader with specified variant.

        Args:
            variant: Optional ModelVariant specifying which variant to use.
                     If None, DEFAULT_VARIANT is used.
        """
        super().__init__(variant)
        self.image_processor = None
        self.model = None

    def load_model(self, dtype_override=None):
        """Load and return the ViT model instance for this instance's variant.

        Args:
            dtype_override: Optional torch.dtype to override the model's default dtype.
                           If not provided, the model will use its default dtype (typically float32).

        Returns:
            torch.nn.Module: The ViT model instance.
        """
        # Get the pretrained model name and source from the instance's variant config
        model_name = self._variant_config.pretrained_model_name
        source = self._variant_config.source

        if source == ModelSource.HUGGING_FACE:
            # Load model from HuggingFace
            model = ViTForImageClassification.from_pretrained(model_name)

        elif source == ModelSource.TORCHVISION:
            # Load model from torchvision
            # Get the weights class name (e.g., "vit_b_16" -> "ViT_B_16_Weights")
            weight_class_name = model_name.upper() + "_Weights"
            weight_class_name = weight_class_name.replace("VIT_", "ViT_")

            # Get the weights class and model function
            weights = getattr(models, weight_class_name).DEFAULT
            model_func = getattr(models, model_name)
            model = model_func(weights=weights)

        model.eval()

        # Store model for potential use in post_processing
        self.model = model

        # Only convert dtype if explicitly requested
        if dtype_override is not None:
            model = model.to(dtype_override)

        return model

    def load_inputs(self, dtype_override=None, batch_size=1):
        """Load and return sample inputs for the ViT model with this instance's variant settings.

        Args:
            dtype_override: Optional torch.dtype to override the inputs' default dtype.
                           If not provided, inputs will use the default dtype (typically float32).
            batch_size: Optional batch size to override the default batch size of 1.

        Returns:
            torch.Tensor: Preprocessed input tensor suitable for ViT.
        """
        # Get the pretrained model name and source from the instance's variant config
        model_name = self._variant_config.pretrained_model_name
        source = self._variant_config.source

        # Get the Image
        image_file = get_file("http://images.cocodataset.org/val2017/000000039769.jpg")
        image = Image.open(image_file).convert("RGB")

        if source == ModelSource.HUGGING_FACE:
            # Initialize image processor if not already done
            if self.image_processor is None:
                self.image_processor = AutoImageProcessor.from_pretrained(
                    model_name, use_fast=True
                )

            # Preprocess image using HuggingFace image processor
            inputs = self.image_processor(
                images=image, return_tensors="pt"
            ).pixel_values

        elif source == ModelSource.TORCHVISION:
            # Get the weights class name for torchvision preprocessing
            weight_class_name = model_name.upper() + "_Weights"
            weight_class_name = weight_class_name.replace("VIT_", "ViT_")

            # Get the weights and use their transforms
            weights = getattr(models, weight_class_name).DEFAULT
            preprocess = weights.transforms()
            inputs = preprocess(image).unsqueeze(0)

        # Replicate tensors for batch size
        inputs = inputs.repeat_interleave(batch_size, dim=0)

        # Only convert dtype if explicitly requested
        if dtype_override is not None:
            inputs = inputs.to(dtype_override)

        return inputs

    def post_processing(self, co_out):
        """Print classification results.

        Args:
            co_out: Output from the compiled model
        """
        source = self._variant_config.source

        if source == ModelSource.HUGGING_FACE:
            logits = co_out[0]
            predicted_class_indices = logits.argmax(-1)

            # Handle both single and batch predictions
            if predicted_class_indices.dim() == 0:  # Single prediction (scalar)
                print(
                    "Predicted class:",
                    self.model.config.id2label[predicted_class_indices.item()],
                )
            else:  # Batch predictions
                for i, idx in enumerate(predicted_class_indices):
                    class_name = self.model.config.id2label[idx.item()]
                    print(f"Batch {i}: Predicted class: {class_name}")

        elif source == ModelSource.TORCHVISION:
            print_compiled_model_results(co_out)
