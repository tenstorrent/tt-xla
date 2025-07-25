# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
RegNet model loader implementation
"""

from typing import Optional
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
from transformers import AutoFeatureExtractor, RegNetForImageClassification
from PIL import Image
from ...tools.utils import get_file


class ModelVariant(StrEnum):
    """Available RegNet model variants."""

    Y_040 = "regnet_y_040"
    Y_064 = "regnet_y_064"
    Y_080 = "regnet_y_080"
    Y_120 = "regnet_y_120"
    Y_160 = "regnet_y_160"
    Y_320 = "regnet_y_320"


class ModelLoader(ForgeModel):
    """RegNet model loader implementation."""

    # Dictionary of available model variants using structured configs
    _VARIANTS = {
        ModelVariant.Y_040: ModelConfig(
            pretrained_model_name="facebook/regnet-y-040",
        ),
        ModelVariant.Y_064: ModelConfig(
            pretrained_model_name="facebook/regnet-y-064",
        ),
        ModelVariant.Y_080: ModelConfig(
            pretrained_model_name="facebook/regnet-y-080",
        ),
        ModelVariant.Y_120: ModelConfig(
            pretrained_model_name="facebook/regnet-y-120",
        ),
        ModelVariant.Y_160: ModelConfig(
            pretrained_model_name="facebook/regnet-y-160",
        ),
        ModelVariant.Y_320: ModelConfig(
            pretrained_model_name="facebook/regnet-y-320",
        ),
    }

    # Default variant to use
    DEFAULT_VARIANT = ModelVariant.Y_040

    def __init__(self, variant: Optional[ModelVariant] = None):
        """Initialize ModelLoader with specified variant.

        Args:
            variant: Optional ModelVariant specifying which variant to use.
                     If None, DEFAULT_VARIANT is used.
        """
        super().__init__(variant)
        self.feature_extractor = None
        self.model = None

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
            model="regnet",
            variant=variant,
            group=ModelGroup.GENERALITY,
            task=ModelTask.CV_IMAGE_CLS,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def load_model(self, dtype_override=None):
        """Load and return the RegNet model instance for this instance's variant.

        Args:
            dtype_override: Optional torch.dtype to override the model's default dtype.
                           If not provided, the model will use its default dtype (typically float32).

        Returns:
            torch.nn.Module: The RegNet model instance.
        """
        # Get the pretrained model name from the instance's variant config
        model_name = self._variant_config.pretrained_model_name

        # Load model from HuggingFace
        model = RegNetForImageClassification.from_pretrained(
            model_name, return_dict=False
        )
        model.eval()

        # Store model for potential use in post_processing
        self.model = model

        # Only convert dtype if explicitly requested
        if dtype_override is not None:
            model = model.to(dtype_override)

        return model

    def load_inputs(self, dtype_override=None, batch_size=1):
        """Load and return sample inputs for the RegNet model with this instance's variant settings.

        Args:
            dtype_override: Optional torch.dtype to override the inputs' default dtype.
                           If not provided, inputs will use the default dtype (typically float32).
            batch_size: Optional batch size to override the default batch size of 1.

        Returns:
            torch.Tensor: Preprocessed input tensor suitable for RegNet.
        """
        # Get the pretrained model name from the instance's variant config
        model_name = self._variant_config.pretrained_model_name

        # Get the Image
        image_file = get_file("http://images.cocodataset.org/val2017/000000039769.jpg")
        image = Image.open(image_file)

        # Initialize feature extractor if not already done
        if self.feature_extractor is None:
            self.feature_extractor = AutoFeatureExtractor.from_pretrained(model_name)

        # Preprocess image
        inputs = self.feature_extractor(images=image, return_tensors="pt").pixel_values

        # Replicate tensors for batch size
        inputs = inputs.repeat_interleave(batch_size, dim=0)

        # Only convert dtype if explicitly requested
        if dtype_override is not None:
            inputs = inputs.to(dtype_override)

        return inputs

    def post_processing(self, co_out):
        """Post-process the model outputs.

        Args:
            co_out: Compiled model outputs

        Returns:
            None: Prints the predicted class
        """

        logits = co_out[0]
        predicted_label = logits.argmax(-1).item()
        print("Predicted class:", self.model.config.id2label[predicted_label])
