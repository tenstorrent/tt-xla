# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Inception model loader implementation
"""

from typing import Optional
from dataclasses import dataclass
import timm
from PIL import Image
from torchvision import transforms
from datasets import load_dataset

from ...tools.utils import VisionPreprocessor, VisionPostprocessor

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
from pytorchcv.model_provider import get_model as ptcv_get_model


@dataclass
class InceptionConfig(ModelConfig):
    """Configuration specific to Inception models"""

    source: ModelSource


class ModelVariant(StrEnum):
    """Available Inception model variants."""

    # TIMM variants
    INCEPTION_V4 = "v4"
    INCEPTION_V4_TF_IN1K = "V4.tf_In1k"

    # OSMR variants
    INCEPTION_V4_OSMR = "v4_OSMR"


class ModelLoader(ForgeModel):
    """Inception model loader implementation."""

    # Dictionary of available model variants using structured configs
    _VARIANTS = {
        # TIMM variants
        ModelVariant.INCEPTION_V4: InceptionConfig(
            pretrained_model_name="inception_v4",
            source=ModelSource.TIMM,
        ),
        ModelVariant.INCEPTION_V4_TF_IN1K: InceptionConfig(
            pretrained_model_name="inception_v4.tf_in1k",
            source=ModelSource.TIMM,
        ),
        # OSMR variants
        ModelVariant.INCEPTION_V4_OSMR: InceptionConfig(
            pretrained_model_name="inceptionv4",
            source=ModelSource.OSMR,
        ),
    }

    # Default variant to use
    DEFAULT_VARIANT = ModelVariant.INCEPTION_V4

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
            model="Inception",
            variant=variant,
            group=ModelGroup.GENERALITY,
            task=ModelTask.CV_IMAGE_CLS,
            source=source,
            framework=Framework.TORCH,
        )

    def load_model(self, *, dtype_override=None, **kwargs):
        """Load and return the Inception model instance for this instance's variant.

        Args:
            dtype_override: Optional torch.dtype to override the model's default dtype.
                           If not provided, the model will use its default dtype (typically float32).

        Returns:
            torch.nn.Module: The Inception model instance.
        """
        # Get the pretrained model name and source from the instance's variant config
        model_name = self._variant_config.pretrained_model_name
        source = self._variant_config.source

        if source == ModelSource.OSMR:
            # Load model using pytorchcv (OSMR)
            model = ptcv_get_model(model_name, pretrained=True)
        elif source == ModelSource.TIMM:
            # Load model using timm
            model = timm.create_model(model_name, pretrained=True)
        else:
            raise ValueError(f"Unsupported Inception source: {source}")

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
        source = self._variant_config.source

        # Handle OSMR variant separately (requires custom preprocessing)
        if source == ModelSource.OSMR:
            if image is None:
                # Load image from HuggingFace dataset
                dataset = load_dataset("huggingface/cats-image")["test"]
                image = dataset[0]["image"].convert("RGB")

            # Convert to PIL if needed
            if isinstance(image, str):
                # If image is a string URL, load from dataset instead
                dataset = load_dataset("huggingface/cats-image")["test"]
                image = dataset[0]["image"].convert("RGB")
            elif not isinstance(image, Image.Image):
                # If it's a tensor or other type, we need to handle it
                # For now, assume it's already processed or raise an error
                raise ValueError(
                    "OSMR variant requires PIL Image input. Please provide a PIL Image or URL."
                )

            preprocess = transforms.Compose(
                [
                    transforms.Resize(299),
                    transforms.CenterCrop(299),
                    transforms.ToTensor(),
                    transforms.Normalize(
                        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                    ),
                ]
            )
            inputs = preprocess(image).unsqueeze(0)

            # Replicate tensors for batch size
            inputs = inputs.repeat_interleave(batch_size, dim=0)

            # Only convert dtype if explicitly requested
            if dtype_override is not None:
                inputs = inputs.to(dtype_override)

            return inputs

        # Standard TIMM preprocessing using VisionPreprocessor
        if self._preprocessor is None:
            model_name = self._variant_config.pretrained_model_name

            self._preprocessor = VisionPreprocessor(
                model_source=source,
                model_name=model_name,
                high_res_size=None,
                weight_class_name_fn=None,
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
        """Load and return sample inputs.

        Args:
            dtype_override: Optional torch.dtype override.
            batch_size: Batch size (default: 1).
            image: Optional input image.

        Returns:
            torch.Tensor: Preprocessed input tensor.
        """
        return self.input_preprocess(
            image=image,
            dtype_override=dtype_override,
            batch_size=batch_size,
        )

    def output_postprocess(self, output):
        """Post-process model outputs.

        Args:
            output: Model output tensor.

        Returns:
            dict: Prediction dictionary with top predictions.
        """
        if self._postprocessor is None:
            model_name = self._variant_config.pretrained_model_name
            source = self._variant_config.source

            self._postprocessor = VisionPostprocessor(
                model_source=source,
                model_name=model_name,
                model_instance=self.model,
            )

        return self._postprocessor.postprocess(output, top_k=1, return_dict=True)
