# SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Deit model loader implementation
"""

from typing import Optional
from dataclasses import dataclass
from transformers import ViTForImageClassification

from ....config import (
    ModelConfig,
    ModelInfo,
    ModelGroup,
    ModelTask,
    ModelSource,
    Framework,
    StrEnum,
)
from ....base import ForgeModel
from ....tools.utils import VisionPreprocessor, VisionPostprocessor


@dataclass
class DeitConfig(ModelConfig):
    """Configuration specific to DeiT models"""

    source: ModelSource
    high_res_size: tuple = (
        None  # None means use default size, otherwise (width, height)
    )


class ModelVariant(StrEnum):
    """Available DeiT model variants."""

    BASE = "base"
    BASE_DISTILLED = "base_distilled"
    SMALL = "small"
    TINY = "tiny"


class ModelLoader(ForgeModel):
    """Deit model loader implementation."""

    # Dictionary of available model variants using structured configs
    _VARIANTS = {
        ModelVariant.BASE: DeitConfig(
            pretrained_model_name="facebook/deit-base-patch16-224",
            source=ModelSource.HUGGING_FACE,
        ),
        ModelVariant.BASE_DISTILLED: DeitConfig(
            pretrained_model_name="facebook/deit-base-distilled-patch16-224",
            source=ModelSource.HUGGING_FACE,
        ),
        ModelVariant.SMALL: DeitConfig(
            pretrained_model_name="facebook/deit-small-patch16-224",
            source=ModelSource.HUGGING_FACE,
        ),
        ModelVariant.TINY: DeitConfig(
            pretrained_model_name="facebook/deit-tiny-patch16-224",
            source=ModelSource.HUGGING_FACE,
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
            model="deit",
            variant=variant,
            group=ModelGroup.GENERALITY,
            task=ModelTask.CV_IMAGE_CLS,
            source=source,
            framework=Framework.TORCH,
        )

    def load_model(self, dtype_override=None):
        """Load and return the Deit model instance for this instance's variant.

        Args:
            dtype_override: Optional torch.dtype to override the model's default dtype.
                           If not provided, the model will use its default dtype (typically float32).

        Returns:
            torch.nn.Module: The Deit model instance.
        """
        # Get the pretrained model name and source from the instance's variant config
        pretrained_model_name = self._variant_config.pretrained_model_name
        source = self._variant_config.source

        if source == ModelSource.HUGGING_FACE:
            # Load pre-trained model from HuggingFace
            model = ViTForImageClassification.from_pretrained(pretrained_model_name)
        else:
            raise ValueError(f"Unsupported source for DeiT: {source}")

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
            torch.Tensor: Preprocessed input tensor (pixel_values).
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

        # For HuggingFace models, VisionPreprocessor returns pixel_values tensor
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
            co_out: Compiled model outputs (prints results if provided).
            framework_model: Original framework model.
            compiled_model: Compiled model.
            inputs: Input images.
            dtype_override: Optional dtype override.

        Returns:
            dict or None: Prediction dict if output provided, else None (prints results).
        """
        if self._postprocessor is None:
            model_name = self._variant_config.pretrained_model_name
            source = self._variant_config.source

            self._postprocessor = VisionPostprocessor(
                model_source=source,
                model_name=model_name,
                model_instance=self.model,
            )

        if output is not None:
            return self._postprocessor.postprocess(output, top_k=1, return_dict=True)

        if co_out is not None:
            if isinstance(co_out, (list, tuple)) and len(co_out) > 0:
                output_tensor = co_out[0]
            else:
                output_tensor = co_out

            self._postprocessor.print_results(
                co_out=output_tensor,
                framework_model=framework_model,
                compiled_model=compiled_model,
                inputs=inputs,
                dtype_override=dtype_override,
            )
        return None
