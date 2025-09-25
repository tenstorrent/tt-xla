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
    AutoImageProcessor,
)
from PIL import Image

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
from ...tools.utils import get_file


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
        self.image_processor = None
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
            model="segformer",
            variant=variant,
            group=ModelGroup.RED
            if variant == ModelVariant.MIT_B0
            else ModelGroup.GENERALITY,
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

        # Store model for potential use in post_processing
        self.model = model

        # Only convert dtype if explicitly requested
        if dtype_override is not None:
            model = model.to(dtype_override)

        return model

    def load_inputs(self, dtype_override=None, batch_size=1):
        """Load and return sample inputs for the Segformer model with this instance's variant settings.

        Args:
            dtype_override: Optional torch.dtype to override the inputs' default dtype.
                           If not provided, inputs will use the default dtype (typically float32).
            batch_size: Optional batch size to override the default batch size of 1.

        Returns:
            torch.Tensor: Preprocessed input tensor suitable for Segformer.
        """
        # Get the pretrained model name from the instance's variant config
        model_name = self._variant_config.pretrained_model_name

        # Get the Image
        image_file = get_file("http://images.cocodataset.org/val2017/000000039769.jpg")
        image = Image.open(image_file)

        # Initialize image processor if not already done
        if self.image_processor is None:
            self.image_processor = AutoImageProcessor.from_pretrained(model_name)

        # Create tokenized inputs
        inputs = self.image_processor(images=image, return_tensors="pt").pixel_values

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
