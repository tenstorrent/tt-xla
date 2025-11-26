# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Segformer model loader implementation for image classification
"""
import torch
from typing import Optional
from transformers import (
    SegformerConfig,
    SegformerForImageClassification,
)

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


class ModelVariant(StrEnum):
    """Available Segformer model variants."""

    MIT_B0 = "mit_b0"
    MIT_B1 = "mit_b1"
    MIT_B2 = "mit_b2"
    MIT_B3 = "mit_b3"
    MIT_B4 = "mit_b4"
    MIT_B5 = "mit_b5"


class ModelLoader(ForgeModel):
    """Segformer model loader implementation."""

    # Dictionary of available model variants using structured configs
    _VARIANTS = {
        ModelVariant.MIT_B0: ModelConfig(
            pretrained_model_name="nvidia/mit-b0",
        ),
        ModelVariant.MIT_B1: ModelConfig(
            pretrained_model_name="nvidia/mit-b1",
        ),
        ModelVariant.MIT_B2: ModelConfig(
            pretrained_model_name="nvidia/mit-b2",
        ),
        ModelVariant.MIT_B3: ModelConfig(
            pretrained_model_name="nvidia/mit-b3",
        ),
        ModelVariant.MIT_B4: ModelConfig(
            pretrained_model_name="nvidia/mit-b4",
        ),
        ModelVariant.MIT_B5: ModelConfig(
            pretrained_model_name="nvidia/mit-b5",
        ),
    }

    # Default variant to use
    DEFAULT_VARIANT = ModelVariant.MIT_B0

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
        return ModelInfo(
            model="segformer",
            variant=variant,
            group=(
                ModelGroup.RED
                if variant == ModelVariant.MIT_B0
                else ModelGroup.GENERALITY
            ),
            task=ModelTask.CV_IMAGE_CLS,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def load_model(self, dtype_override=None):
        """Load and return the Segformer model instance for this instance's variant.

        Args:
            dtype_override: Optional torch.dtype to override the model's default dtype.
                           If not provided, the model will use its default dtype (typically float32).

        Returns:
            torch.nn.Module: The Segformer model instance.
        """
        # Get the pretrained model name from the instance's variant config
        model_name = self._variant_config.pretrained_model_name

        # Load configuration
        config = SegformerConfig.from_pretrained(model_name)
        config_dict = config.to_dict()
        config = SegformerConfig(**config_dict)

        # Load model from HuggingFace
        model = SegformerForImageClassification.from_pretrained(
            model_name, config=config
        )
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
            # All segformer models are from HuggingFace
            source = ModelSource.HUGGING_FACE

            self._preprocessor = VisionPreprocessor(
                model_source=source,
                model_name=model_name,
            )

            if hasattr(self, "model") and self.model is not None:
                self._preprocessor.set_cached_model(self.model)

        model_for_config = None
        # Segformer models are from HuggingFace, not TIMM, so model_for_config is not needed

        return self._preprocessor.preprocess(
            image=image,
            dtype_override=dtype_override,
            batch_size=batch_size,
            model_for_config=model_for_config,
        )

    def load_inputs(self, dtype_override=None, batch_size=1, image=None):
        """Load and return sample inputs (backward compatibility wrapper for input_preprocess).

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

    def output_postprocess(
        self,
        output=None,
        co_out=None,
        framework_model=None,
        compiled_model=None,
        inputs=None,
        dtype_override=None,
    ):
        """Post-process model outputs.

        Args:
            output: Model output tensor (returns dict if provided).
            co_out: Compiled model outputs (legacy, prints results).
            framework_model: Original framework model (legacy).
            compiled_model: Compiled model (legacy).
            inputs: Input images (legacy).
            dtype_override: Optional dtype override (legacy).

        Returns:
            dict or None: Prediction dict if output provided, else None (prints results).
        """
        if self._postprocessor is None:
            model_name = self._variant_config.pretrained_model_name
            # All segformer models are from HuggingFace
            source = ModelSource.HUGGING_FACE

            self._postprocessor = VisionPostprocessor(
                model_source=source,
                model_name=model_name,
                model_instance=self.model,
            )

        # New usage: return dict from output tensor
        if output is not None:
            return self._postprocessor.postprocess(output, top_k=1, return_dict=True)

        # Legacy usage: print results (backward compatibility)
        self._postprocessor.print_results(
            co_out=co_out,
            framework_model=framework_model,
            compiled_model=compiled_model,
            inputs=inputs,
            dtype_override=dtype_override,
        )
        return None

    def post_processing(self, co_out):
        """Post-process the model outputs (backward compatibility wrapper).

        Args:
            co_out: Compiled model outputs

        Returns:
            None: Prints the predicted class
        """
        self.output_postprocess(co_out=co_out)
