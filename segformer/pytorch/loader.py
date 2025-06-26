# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Segformer model loader implementation for question answering
"""
import torch
from transformers import (
    SegformerConfig,
    SegformerForImageClassification,
)
from PIL import Image

from transformers import AutoImageProcessor
from ...config import (
    ModelInfo,
    ModelGroup,
    ModelTask,
    ModelSource,
    Framework,
)
from ...base import ForgeModel
from ...tools.utils import get_file


class ModelLoader(ForgeModel):
    @classmethod
    def _get_model_info(cls, variant_name: str = None):
        """Get model information for dashboard and metrics reporting.

        Args:
            variant_name: Optional variant name string. If None, uses 'base'.

        Returns:
            ModelInfo: Information about the model and variant
        """
        if variant_name is None:
            variant_name = "base"
        return ModelInfo(
            model="segformer",
            variant=variant_name,
            group=ModelGroup.PRIORITY,
            task=ModelTask.CV_IMAGE_SEG,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def __init__(self, variant=None):
        """Initialize ModelLoader with specified variant.

        Args:
            variant: Optional string specifying which variant to use.
                     If None, DEFAULT_VARIANT is used.
        """
        super().__init__(variant)

        # Configuration parameters
        self.model_name = "nvidia/mit-b0"

    def load_model(self, dtype_override=None):
        """Load a Segformer model from Hugging Face."""
        config = SegformerConfig.from_pretrained(self.model_name)
        config_dict = config.to_dict()
        config_dict["return_dict"] = False
        config = SegformerConfig(**config_dict)
        model = SegformerForImageClassification.from_pretrained(
            self.model_name, config=config
        )
        model.eval()
        self.model = model

        # Only convert dtype if explicitly requested
        if dtype_override is not None:
            model = model.to(dtype_override)

        return model

    def load_inputs(self, dtype_override=None):
        """Generate sample inputs for Segformer models."""
        # Get the Image
        image_file = get_file("http://images.cocodataset.org/val2017/000000039769.jpg")
        image = Image.open(image_file)

        # Initialize tokenizer
        image_processor = AutoImageProcessor.from_pretrained(self.model_name)

        # Create tokenized inputs
        inputs = image_processor(images=image, return_tensors="pt").pixel_values

        # Only convert dtype if explicitly requested
        if dtype_override is not None:
            inputs = inputs.to(dtype_override)

        return inputs

    def post_processing(self, co_out):
        logits = co_out[0]
        predicted_label = logits.argmax(-1).item()
        print("Predicted class: ", self.model.config.id2label[predicted_label])
