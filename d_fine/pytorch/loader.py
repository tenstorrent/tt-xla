# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
D-Fine model loader implementation for object detection
"""
import torch
from transformers.image_utils import load_image
from transformers import DFineForObjectDetection, AutoImageProcessor
from datasets import load_dataset
from typing import Optional

from ...base import ForgeModel
from ...config import (
    ModelConfig,
    ModelInfo,
    ModelGroup,
    ModelTask,
    ModelSource,
    Framework,
    StrEnum,
)


class ModelVariant(StrEnum):
    """Available D-Fine model variants."""

    NANO = "Nano"
    SMALL = "Small"
    MEDIUM = "Medium"
    LARGE = "Large"
    XLARGE = "Xlarge"


class ModelLoader(ForgeModel):
    """D-Fine model loader implementation for object detection tasks."""

    # Dictionary of available model variants
    _VARIANTS = {
        ModelVariant.NANO: ModelConfig(
            pretrained_model_name="ustc-community/dfine-nano-coco",
        ),
        ModelVariant.SMALL: ModelConfig(
            pretrained_model_name="ustc-community/dfine-small-coco",
        ),
        ModelVariant.MEDIUM: ModelConfig(
            pretrained_model_name="ustc-community/dfine-medium-coco",
        ),
        ModelVariant.LARGE: ModelConfig(
            pretrained_model_name="ustc-community/dfine-large-coco",
        ),
        ModelVariant.XLARGE: ModelConfig(
            pretrained_model_name="ustc-community/dfine-xlarge-coco",
        ),
    }

    # Default variant to use
    DEFAULT_VARIANT = ModelVariant.SMALL

    def __init__(self, variant: Optional[ModelVariant] = None):
        """Initialize ModelLoader with specified variant.

        Args:
            variant: Optional ModelVariant specifying which variant to use.
                     If None, DEFAULT_VARIANT is used.
        """
        super().__init__(variant)
        self.processor = None
        self.image = None
        self.model = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        """Implementation method for getting model info with validated variant.

        Args:
            variant: Optional ModelVariant specifying which variant to use.
                     If None, DEFAULT_VARIANT is used.

        Returns:
            ModelInfo: Information about the model and variant
        """
        return ModelInfo(
            model="D-FINE",
            variant=variant,
            group=ModelGroup.GENERALITY,
            task=ModelTask.CV_OBJECT_DET,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def _load_processor(self):
        """Load processor for the current variant.

        Returns:
            The loaded processor instance
        """

        # Initialize processor
        self.processor = AutoImageProcessor.from_pretrained(
            self._variant_config.pretrained_model_name
        )

        return self.processor

    def load_model(self, *, dtype_override=None, **kwargs):
        """Load and return the D-Fine model instance for this instance's variant.

        Args:
            dtype_override: Optional torch.dtype to override the model's default dtype.
                           If not provided, the model will use its default dtype (typically float32).

        Returns:
            torch.nn.Module: The D-Fine model instance for object detection.
        """
        # Get the pretrained model name from the instance's variant config
        pretrained_model_name = self._variant_config.pretrained_model_name

        # Ensure processor is loaded
        if self.processor is None:
            self._load_processor()

        model_kwargs = {}
        if dtype_override is not None:
            model_kwargs["torch_dtype"] = dtype_override
        model_kwargs |= kwargs

        # Load pre-trained model from HuggingFace
        model = DFineForObjectDetection.from_pretrained(
            pretrained_model_name, **model_kwargs
        )
        self.model = model  # Store model for later use
        return model

    def load_inputs(self, dtype_override=None, batch_size=1):
        """Load and return sample inputs for the D-Fine model with this instance's variant settings.

        Args:
            dtype_override: Optional torch.dtype to override the model inputs' default dtype.
            batch_size: Optional batch size to override the default batch size of 1.

        Returns:
            dict: Input tensors (pixel values) that can be fed to the model.
        """
        # Ensure processor is initialized
        if self.processor is None:
            self._load_processor()

        # Load image from HuggingFace dataset
        dataset = load_dataset("huggingface/cats-image")["test"]
        self.image = dataset[0]["image"]
        inputs = self.processor(images=self.image, return_tensors="pt")

        if dtype_override is not None:
            inputs = inputs.to(dtype_override)

        # Add batch dimension if batch_size
        for key in inputs:
            if torch.is_tensor(inputs[key]):
                inputs[key] = inputs[key].repeat_interleave(batch_size, dim=0)

        return inputs

    def decode_output(self, outputs, threshold=0.5):
        """Helper method to decode model outputs into human-readable object detection results.

        Args:
            outputs: Model output from a forward pass
            threshold: Optional confidence threshold for filtering detections (default: 0.5)

        Returns:
            str: Formatted detection results with labels, scores, and bounding boxes
        """
        if self.processor is None:
            self._load_processor()

        if self.image is None:
            # Load image from HuggingFace dataset if not already loaded
            dataset = load_dataset("huggingface/cats-image")["test"]
            self.image = dataset[0]["image"]

        # Post-process the model outputs
        results = self.processor.post_process_object_detection(
            outputs,
            target_sizes=[(self.image.height, self.image.width)],
            threshold=threshold,
        )

        # Format the results
        detection_strings = []
        for result in results:
            for score, label_id, box in zip(
                result["scores"], result["labels"], result["boxes"]
            ):
                score, label = score.item(), label_id.item()
                box = [round(i, 2) for i in box.tolist()]

                # Get model to access id2label mapping
                if hasattr(self, "model") and self.model is not None:
                    label_name = self.model.config.id2label[label]
                else:
                    # Fallback if model not stored
                    label_name = f"class_{label}"

                detection_strings.append(f"{label_name}: {score:.2f} {box}")

        return (
            "\n".join(detection_strings) if detection_strings else "No detections found"
        )
