# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
ViLT model loader implementation
"""

from transformers import ViltForQuestionAnswering, ViltProcessor
from ...config import (
    ModelInfo,
    ModelGroup,
    ModelTask,
    ModelSource,
    Framework,
)
from ...base import ForgeModel
from PIL import Image
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
            model="vilt",
            variant=variant_name,
            group=ModelGroup.GENERALITY,
            task=ModelTask.MM_VISUAL_QA,
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
        self.model_name = "dandelin/vilt-b32-finetuned-vqa"
        self.text = "How many cats are there?"
        self.processor = None

    def load_model(self, dtype_override=None):
        """Load a ViLT model from Hugging Face."""

        # Initialize processor first with default or overridden dtype
        processor_kwargs = {}
        if dtype_override is not None:
            processor_kwargs["torch_dtype"] = dtype_override

        self.processor = ViltProcessor.from_pretrained(
            self.model_name, **processor_kwargs
        )

        # Load pre-trained model from HuggingFace
        model_kwargs = {}
        if dtype_override is not None:
            model_kwargs["torch_dtype"] = dtype_override

        model = ViltForQuestionAnswering.from_pretrained(
            self.model_name, return_dict=False, **model_kwargs
        )
        model.eval()
        return model

    def load_inputs(self):
        """Generate sample inputs for ViLT model."""

        # Ensure processor is initialized
        if not hasattr(cls, "processor"):
            self.load_model()  # This will initialize the processor

        # Get the Image
        image_file = get_file("http://images.cocodataset.org/val2017/000000039769.jpg")
        image = Image.open(image_file)
        inputs = self.processor(image, self.text, return_tensors="pt")

        return inputs
