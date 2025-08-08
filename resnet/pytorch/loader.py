# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Resnet model loader implementation for question answering
"""
import torch

from transformers import ResNetForImageClassification
from ...tools.utils import print_compiled_model_results
from transformers import AutoImageProcessor
from tabulate import tabulate

from ...config import (
    ModelInfo,
    ModelGroup,
    ModelTask,
    ModelSource,
    Framework,
)
from ...base import ForgeModel


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
            model="resnet",
            variant=variant_name,
            group=ModelGroup.RED,
            task=ModelTask.CV_IMAGE_CLS,
            source=ModelSource.TORCH_HUB,
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
        self.model_name = "microsoft/resnet-50"
        self.input_shape = (3, 224, 224)

    def load_model(self, dtype_override=None):
        """Load a Resnet model from Hugging Face."""
        model = ResNetForImageClassification.from_pretrained(
            self.model_name, return_dict=False
        )

        model.eval()

        # Only convert dtype if explicitly requested
        if dtype_override is not None:
            model = model.to(dtype_override)

        return model

    def load_inputs(self, dtype_override=None):
        """Generate sample inputs for Resnet models."""

        # Create a random input tensor with the correct shape, using default dtype
        inputs = torch.rand(1, *self.input_shape)

        # Only convert dtype if explicitly requested
        if dtype_override is not None:
            inputs = inputs.to(dtype_override)

        return inputs

    def run_and_print_results(self, framework_model, compiled_model, inputs):
        """
        Runs inference using both a framework model and a compiled model on a list of input images,
        then prints the results in a formatted table.

        Args:
            framework_model: The original framework-based model.
            compiled_model: The compiled version of the model.
            inputs: A list of images to process and classify.
        """
        label_dict = framework_model.config.id2label
        processor = AutoImageProcessor.from_pretrained("microsoft/resnet-50")

        results = []
        for i, image in enumerate(inputs):
            processed_inputs = processor(image, return_tensors="pt")["pixel_values"].to(
                torch.bfloat16
            )

            cpu_logits = framework_model(processed_inputs)[0]
            cpu_conf, cpu_idx = cpu_logits.softmax(-1).max(-1)
            cpu_pred = label_dict.get(cpu_idx.item(), "Unknown")

            tt_logits = compiled_model(processed_inputs)[0]
            tt_conf, tt_idx = tt_logits.softmax(-1).max(-1)
            tt_pred = label_dict.get(tt_idx.item(), "Unknown")

            results.append([i + 1, cpu_pred, cpu_conf.item(), tt_pred, tt_conf.item()])

        print(
            tabulate(
                results,
                headers=[
                    "Example",
                    "CPU Prediction",
                    "CPU Confidence",
                    "Compiled Prediction",
                    "Compiled Confidence",
                ],
                tablefmt="grid",
            )
        )
