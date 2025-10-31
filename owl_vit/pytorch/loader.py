# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
OWL-ViT model loader implementation for object detection.
"""
import torch
from transformers import OwlViTProcessor, OwlViTForObjectDetection
from transformers.models.owlvit.modeling_owlvit import OwlViTObjectDetectionOutput
from typing import Optional
from PIL import Image

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
from ...tools.utils import get_file


class ModelVariant(StrEnum):
    """Available OWL-ViT model variants for object detection."""

    BASE_PATCH32 = "base_patch32"


class ModelLoader(ForgeModel):
    """OWL-ViT model loader implementation for object detection tasks."""

    # Dictionary of available model variants using structured configs
    _VARIANTS = {
        ModelVariant.BASE_PATCH32: ModelConfig(
            pretrained_model_name="google/owlvit-base-patch32",
        ),
    }

    # Default variant to use
    DEFAULT_VARIANT = ModelVariant.BASE_PATCH32

    def __init__(self, variant: Optional[ModelVariant] = None):
        """Initialize ModelLoader with specified variant.

        Args:
            variant: Optional ModelVariant specifying which variant to use.
                     If None, DEFAULT_VARIANT is used.
        """
        super().__init__(variant)
        self.processor = None
        self.image = None
        self.text_labels = None

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
            model="owl_vit_detection",
            variant=variant,
            group=ModelGroup.RED,
            task=ModelTask.CV_OBJECT_DET,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def _load_processor(self):
        """Load processor for the current variant.

        Returns:
            The loaded processor instance
        """
        # Load the processor
        self.processor = OwlViTProcessor.from_pretrained(
            self._variant_config.pretrained_model_name
        )

        return self.processor

    def load_model(self, dtype_override=None):
        """Load and return the OWL-ViT model instance for this instance's variant.

        Args:
            dtype_override: Optional torch.dtype to override the model's default dtype.
                           If not provided, the model will use its default dtype (typically float32).

        Returns:
            torch.nn.Module: The OWL-ViT model instance for object detection.
        """
        # Get the pretrained model name from the instance's variant config
        pretrained_model_name = self._variant_config.pretrained_model_name

        model_kwargs = {"return_dict": False}

        # Load the model with dtype override if specified
        if dtype_override is not None:
            model_kwargs["torch_dtype"] = dtype_override

        model = OwlViTForObjectDetection.from_pretrained(
            pretrained_model_name, **model_kwargs
        )
        model.eval()

        return model

    def load_inputs(self, dtype_override=None, batch_size=1):
        """Load and return sample inputs for the OWL-ViT model with this instance's variant settings.

        Args:
            dtype_override: Optional torch.dtype to override the model inputs' default dtype.
            batch_size: Batch size for the inputs.

        Returns:
            dict: Input tensors that can be fed to the model.
        """
        # Ensure processor is initialized
        if self.processor is None:
            self._load_processor()

        # Get the Image
        image_file = get_file("http://images.cocodataset.org/val2017/000000039769.jpg")
        self.image = Image.open(image_file)

        # Define text labels for object detection
        self.text_labels = [["a photo of a cat", "a photo of a dog"]]

        # Process both text and images
        inputs = self.processor(
            text=self.text_labels, images=self.image, return_tensors="pt"
        )

        # Handle batch size
        for key in inputs:
            if torch.is_tensor(inputs[key]):
                inputs[key] = inputs[key].repeat_interleave(batch_size, dim=0)

        # Convert the input dtype to dtype_override if specified
        if dtype_override is not None:
            inputs["pixel_values"] = inputs["pixel_values"].to(dtype_override)

        return inputs

    def post_process(self, outputs, threshold=0.1):
        """Post-process OWL-ViT model outputs to extract detection results.

        Args:
            outputs: Raw model output tuple (logits, pred_boxes) from OWL-ViT forward pass.
            threshold: Confidence threshold for filtering detections (default: 0.1).

        Returns:
            list: Post-processed detection results with boxes, scores, and text labels.
        """
        # Ensure processor and image are initialized
        if self.processor is None:
            self._load_processor()

        if self.image is None:
            # Load image if not already loaded
            image_file = get_file(
                "http://images.cocodataset.org/val2017/000000039769.jpg"
            )
            self.image = Image.open(image_file)

        if self.text_labels is None:
            # Use text labels if not already set
            self.text_labels = [["a photo of a cat", "a photo of a dog"]]

        # Create OWL-ViT object detection output from model outputs
        owl_vit_outputs = OwlViTObjectDetectionOutput(
            logits=outputs[0], pred_boxes=outputs[1]
        )

        # Target image sizes (height, width) to rescale box predictions [batch_size, 2]
        target_sizes = torch.tensor([(self.image.height, self.image.width)])

        # Convert outputs (bounding boxes and class logits) to Pascal VOC format (xmin, ymin, xmax, ymax)
        results = self.processor.post_process_grounded_object_detection(
            outputs=owl_vit_outputs,
            target_sizes=target_sizes,
            threshold=threshold,
            text_labels=self.text_labels,
        )

        # Retrieve predictions for the image for the corresponding text queries
        result = results[0]
        boxes, scores, text_labels = (
            result["boxes"],
            result["scores"],
            result["text_labels"],
        )
        for box, score, text_label in zip(boxes, scores, text_labels):
            box = [round(i, 2) for i in box.tolist()]
            print(
                f"Detected {text_label} with confidence {round(score.item(), 3)} at location {box}"
            )
