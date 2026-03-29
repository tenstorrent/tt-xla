# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
ViT model loader implementation
"""

from transformers import ViTForImageClassification
from torchvision import models
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
from ...tools.utils import VisionPreprocessor, VisionPostprocessor
from datasets import load_dataset


@dataclass
class ViTConfig(ModelConfig):
    """Configuration specific to ViT models"""

    source: ModelSource


class ModelVariant(StrEnum):
    """Available ViT model variants."""

    # HuggingFace variants
    TINY = "Tiny"
    BASE = "Base"
    BASE_PATCH16_384 = "Base_Patch16_384"
    LARGE = "Large"
    BASE_PATCH32_384 = "Base_Patch32_384"
    BASE_OXFORD_IIIT_PETS = "Base_Oxford_IIIT_Pets"

    # Torchvision variants
    VIT_B_16 = "B_16"
    VIT_B_32 = "B_32"
    VIT_L_16 = "L_16"
    VIT_L_32 = "L_32"
    VIT_H_14 = "H_14"

    # HuggingFace fine-tuned variants
    AI_IMAGE_DETECTOR = "AI_Image_Detector"

    # TIMM variants
    VIT_BASE_PATCH14_DINOV2_LVD142M = "Base_Patch14_DINOv2_LVD142M"
    VIT_BASE_PATCH16_224_AUGREG_IN1K = "Base_Patch16_224_AugReg_IN1K"
    VIT_BASE_PATCH16_224_AUGREG_IN21K = "Base_Patch16_224_AugReg_IN21K"
    VIT_BASE_PATCH16_384_AUGREG_IN21K_FT_IN1K = "Base_Patch16_384_AugReg_IN21K_FT_IN1K"
    VIT_BASE_PATCH32_CLIP_224_LAION2B_E16 = "Base_Patch32_CLIP_224_LAION2B_E16"


class ModelLoader(ForgeModel):
    """ViT model loader implementation."""

    # Dictionary of available model variants using structured configs
    _VARIANTS = {
        # HuggingFace variants
        ModelVariant.TINY: ViTConfig(
            pretrained_model_name="WinKawaks/vit-tiny-patch16-224",
            source=ModelSource.HUGGING_FACE,
        ),
        ModelVariant.BASE: ViTConfig(
            pretrained_model_name="google/vit-base-patch16-224",
            source=ModelSource.HUGGING_FACE,
        ),
        ModelVariant.BASE_PATCH16_384: ViTConfig(
            pretrained_model_name="google/vit-base-patch16-384",
            source=ModelSource.HUGGING_FACE,
        ),
        ModelVariant.LARGE: ViTConfig(
            pretrained_model_name="google/vit-large-patch16-224",
            source=ModelSource.HUGGING_FACE,
        ),
        # HuggingFace fine-tuned variants
        ModelVariant.AI_IMAGE_DETECTOR: ViTConfig(
            pretrained_model_name="umm-maybe/AI-image-detector",
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
        # TIMM variants
        ModelVariant.VIT_BASE_PATCH14_DINOV2_LVD142M: ViTConfig(
            pretrained_model_name="vit_base_patch14_dinov2.lvd142m",
            source=ModelSource.TIMM,
        ),
        ModelVariant.VIT_BASE_PATCH16_224_AUGREG_IN1K: ViTConfig(
            pretrained_model_name="vit_base_patch16_224.augreg_in1k",
            source=ModelSource.TIMM,
        ),
        ModelVariant.VIT_BASE_PATCH16_224_AUGREG_IN21K: ViTConfig(
            pretrained_model_name="vit_base_patch16_224.augreg_in21k",
            source=ModelSource.TIMM,
        ),
        ModelVariant.VIT_BASE_PATCH16_384_AUGREG_IN21K_FT_IN1K: ViTConfig(
            pretrained_model_name="vit_base_patch16_384.augreg_in21k_ft_in1k",
            source=ModelSource.TIMM,
        ),
        ModelVariant.VIT_BASE_PATCH32_CLIP_224_LAION2B_E16: ViTConfig(
            pretrained_model_name="vit_base_patch32_clip_224.laion2b_e16",
            source=ModelSource.TIMM,
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

        # Determine model group
        if variant == ModelVariant.TINY:
            group = ModelGroup.VULCAN
        elif variant == ModelVariant.BASE:
            group = ModelGroup.RED
        elif variant == ModelVariant.AI_IMAGE_DETECTOR:
            group = ModelGroup.VULCAN
        elif cls._VARIANTS[variant].source == ModelSource.TIMM:
            group = ModelGroup.VULCAN
        else:
            group = ModelGroup.GENERALITY

        # Determine task based on variant
        if variant == ModelVariant.VIT_BASE_PATCH16_224_ORIG_IN21K:
            task = ModelTask.CV_IMAGE_FE
        else:
            task = ModelTask.CV_IMAGE_CLS

        return ModelInfo(
            model="ViT",
            variant=variant,
            group=group,
            task=task,
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
        self.model = None
        self._preprocessor = None
        self._postprocessor = None

    def load_model(self, *, dtype_override=None, **kwargs):
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

        if source == ModelSource.TIMM:
            # Load model from TIMM
            import timm

            model = timm.create_model(model_name, pretrained=True)

        elif source == ModelSource.HUGGING_FACE:
            # Load model from HuggingFace
            model = ViTForImageClassification.from_pretrained(model_name, **kwargs)

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

        # Store model for potential use in input preprocessing and postprocessing
        self.model = model

        # Update preprocessor with cached model (for TIMM models)
        if self._preprocessor is not None:
            self._preprocessor.set_cached_model(model)

        # Update postprocessor with model instance (for HuggingFace models)
        if self._postprocessor is not None:
            self._postprocessor.set_model_instance(model)

        # Only convert dtype if explicitly requested
        if dtype_override is not None:
            model = model.to(dtype_override)

        return model

    def input_preprocess(self, dtype_override=None, batch_size=1, image=None):
        """Preprocess input image(s) and return model-ready input tensor.

        Args:
            dtype_override: Optional torch.dtype override (default: float32).
            batch_size: Batch size (ignored if image is a list).
            image: PIL Image, URL string, tensor, list of images/URLs, or None (uses default COCO image).

        Returns:
            torch.Tensor: Preprocessed input tensor.
        """
        if self._preprocessor is None:
            model_name = self._variant_config.pretrained_model_name
            source = self._variant_config.source

            def weight_class_name_fn(name: str) -> str:
                return name.upper().replace("VIT_", "ViT_") + "_Weights"

            self._preprocessor = VisionPreprocessor(
                model_source=source,
                model_name=model_name,
                weight_class_name_fn=(
                    weight_class_name_fn if source == ModelSource.TORCHVISION else None
                ),
                image_processor_kwargs=(
                    {"use_fast": True} if source == ModelSource.HUGGING_FACE else None
                ),
            )

            if hasattr(self, "model") and self.model is not None:
                self._preprocessor.set_cached_model(self.model)

        model_for_config = None
        if self._variant_config.source == ModelSource.TIMM:
            if hasattr(self, "model") and self.model is not None:
                model_for_config = self.model

        return self._preprocessor.preprocess(
            image=image,
            dtype_override=dtype_override,
            batch_size=batch_size,
            model_for_config=model_for_config,
        )

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
        if self._postprocessor is None:
            model_name = self._variant_config.pretrained_model_name
            source = self._variant_config.source

            def weight_class_name_fn(name: str) -> str:
                return name.upper().replace("VIT_", "ViT_") + "_Weights"

            self._postprocessor = VisionPostprocessor(
                model_source=source,
                model_name=model_name,
                model_instance=self.model,
            )

        return self._postprocessor.postprocess(output, top_k=top_k, return_dict=True)
