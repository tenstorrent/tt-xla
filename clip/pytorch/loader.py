# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
CLIP model loader implementation for image-text similarity.
"""
import torch
from transformers import CLIPProcessor, CLIPModel
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
    """Available CLIP model variants for image-text similarity."""

    BASE_PATCH16 = "base_patch16"
    BASE_PATCH32 = "base_patch32"
    LARGE_PATCH14 = "large_patch14"
    LARGE_PATCH14_336 = "large_patch14_336"


class ModelLoader(ForgeModel):
    """CLIP model loader implementation for image-text similarity tasks."""

    # Dictionary of available model variants using structured configs
    _VARIANTS = {
        ModelVariant.BASE_PATCH16: ModelConfig(
            pretrained_model_name="openai/clip-vit-base-patch16",
        ),
        ModelVariant.BASE_PATCH32: ModelConfig(
            pretrained_model_name="openai/clip-vit-base-patch32",
        ),
        ModelVariant.LARGE_PATCH14: ModelConfig(
            pretrained_model_name="openai/clip-vit-large-patch14",
        ),
        ModelVariant.LARGE_PATCH14_336: ModelConfig(
            pretrained_model_name="openai/clip-vit-large-patch14-336",
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

        # Configuration parameters
        self.processor = None
        self.text_prompts = None

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
            model="clip",
            variant=variant,
            group=ModelGroup.RED
            if variant == ModelVariant.BASE_PATCH16
            else ModelGroup.GENERALITY,
            task=ModelTask.MM_IMAGE_TEXT_SIM,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def _load_processor(self):
        """Load processor for the current variant.

        Returns:
            The loaded processor instance
        """
        # Load the processor
        self.processor = CLIPProcessor.from_pretrained(
            self._variant_config.pretrained_model_name
        )

        return self.processor

    def load_model(self, dtype_override=None):
        """Load and return the CLIP model instance for this instance's variant.

        Args:
            dtype_override: Optional torch.dtype to override the model's default dtype.
                           If not provided, the model will use its default dtype (typically float32).

        Returns:
            torch.nn.Module: The CLIP model instance for image-text similarity.
        """
        # Get the pretrained model name from the instance's variant config
        pretrained_model_name = self._variant_config.pretrained_model_name

        model_kwargs = {"return_dict": False}

        # Load the model with dtype override if specified
        if dtype_override is not None:
            model_kwargs["torch_dtype"] = dtype_override

        model = CLIPModel.from_pretrained(pretrained_model_name, **model_kwargs)
        model.eval()

        return model

    def load_inputs(self, dtype_override=None, batch_size=1):
        """Load and return sample inputs for the CLIP model with this instance's variant settings.

        Args:
            dtype_override: Optional torch.dtype to override the model's default dtype.
                           If not provided, the model will use its default dtype (typically float32).
            batch_size: Optional batch size to override the default batch size of 1.

        Returns:
            dict: Input tensors, pixel values and attention masks that can be fed to the model.
        """
        # Ensure processor is initialized
        if self.processor is None:
            self._load_processor()

        # Get the Image
        image_file = get_file("http://images.cocodataset.org/val2017/000000039769.jpg")
        image = Image.open(image_file)

        # Define text prompts for image-text similarity
        self.text_prompts = ["a photo of a cat", "a photo of a dog"]

        # Process both text and images
        inputs = self.processor(
            text=self.text_prompts, images=image, return_tensors="pt", padding=True
        )

        # Replicate tensors for batch size
        for key in inputs:
            if torch.is_tensor(inputs[key]):
                inputs[key] = inputs[key].repeat_interleave(batch_size, dim=0)

        # Convert the input dtype to dtype_override if specified
        if dtype_override is not None:
            inputs["pixel_values"] = inputs["pixel_values"].to(dtype_override)

        return inputs

    def post_process(self, outputs):
        """Post-process CLIP model outputs to extract similarity scores.

        Args:
            outputs: Raw model output

        Returns:
            dict: Post-processed similarity results with probabilities for each text prompt.
        """
        # Ensure text prompts are initialized
        if self.text_prompts is None:
            # Use default text prompts if not already set
            self.text_prompts = ["a photo of a cat", "a photo of a dog"]

        # Extract logits_per_image from outputs
        logits_per_image = outputs[0]  # Image-Text similarity score
        probs = logits_per_image.softmax(
            dim=1
        )  # Softmax to get the label probabilities

        # Print results
        for i, text in enumerate(self.text_prompts):
            print(f"Probability of '{text}':", probs[0, i].item())

    def unpack_forward_output(self, fwd_output):
        """Unpack forward pass output to extract a differentiable tensor.

        The CLIP model (with return_dict=False) returns a tuple containing:
        - logits_per_image: Image-text similarity scores
        - logits_per_text: Text-image similarity scores
        - text_embeds: Text embeddings
        - image_embeds: Image embeddings
        - text_model_output: Detailed text model outputs
        - vision_model_output: Detailed vision model outputs

        For training, we concatenate the main output tensors to create a single
        differentiable tensor for backpropagation.

        Args:
            fwd_output: Output from the model's forward pass (tuple)

        Returns:
            torch.Tensor: Concatenated flattened outputs for backward pass
        """
        if isinstance(fwd_output, tuple):
            tensors = []
            for item in fwd_output:
                if isinstance(item, torch.Tensor):
                    tensors.append(item.flatten())
                elif hasattr(item, "last_hidden_state"):
                    # Handle model output objects with last_hidden_state
                    tensors.append(item.last_hidden_state.flatten())
            if tensors:
                return torch.cat(tensors, dim=0)
        return fwd_output
