# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Ghostnet model loader implementation
"""

from typing import Optional
from dataclasses import dataclass
import timm

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
from ...tools.utils import VisionPreprocessor, VisionPostprocessor


@dataclass
class GhostNetConfig(ModelConfig):
    """Configuration specific to GhostNet models"""

    source: ModelSource
    high_res_size: tuple = (
        None  # None means use default size, otherwise (width, height)
    )


class ModelVariant(StrEnum):
    """Available GhostNet model variants."""

    GHOSTNET_100 = "ghostnet_100"
    GHOSTNET_100_IN1K = "ghostnet_100.in1k"
    GHOSTNETV2_100_IN1K = "ghostnetv2_100.in1k"


class ModelLoader(ForgeModel):
    """GhostNet model loader implementation."""

    # Dictionary of available model variants using structured configs
    _VARIANTS = {
        ModelVariant.GHOSTNET_100: GhostNetConfig(
            pretrained_model_name="ghostnet_100",
            source=ModelSource.TIMM,
        ),
        ModelVariant.GHOSTNET_100_IN1K: GhostNetConfig(
            pretrained_model_name="ghostnet_100.in1k",
            source=ModelSource.TIMM,
        ),
        ModelVariant.GHOSTNETV2_100_IN1K: GhostNetConfig(
            pretrained_model_name="ghostnetv2_100.in1k",
            source=ModelSource.TIMM,
        ),
    }

    # Default variant to use
    DEFAULT_VARIANT = ModelVariant.GHOSTNET_100

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
            model="ghostnet",
            variant=variant,
            group=ModelGroup.GENERALITY,
            task=ModelTask.CV_IMAGE_CLS,
            source=source,
            framework=Framework.TORCH,
        )

    def load_model(self, dtype_override=None):
        """Load pretrained GhostNet model for this instance's variant.

        Args:
            dtype_override: Optional torch.dtype to override the model's default dtype.
                           If not provided, the model will use its default dtype (typically float32).

        Returns:
            torch.nn.Module: The GhostNet model instance.
        """
        # Get the pretrained model name and source from the instance's variant config
        model_name = self._variant_config.pretrained_model_name
        source = self._variant_config.source

        if source == ModelSource.TIMM:
            # Load model using timm
            model = timm.create_model(model_name, pretrained=True)
        else:
            raise ValueError(f"Unsupported source for GhostNet: {source}")

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
            high_res_size = self._variant_config.high_res_size

            self._preprocessor = VisionPreprocessor(
                model_source=source,
                model_name=model_name,
                high_res_size=high_res_size,
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
            dict: Prediction dict with top predictions.
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
