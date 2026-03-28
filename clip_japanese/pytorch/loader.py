# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
CLIP Japanese model loader implementation for image-text similarity.
"""
import torch
from transformers import AutoImageProcessor, AutoModel, AutoTokenizer
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
from datasets import load_dataset


class ModelVariant(StrEnum):
    """Available CLIP Japanese model variants for image-text similarity."""

    BASE = "Base"


class ModelLoader(ForgeModel):
    """CLIP Japanese model loader implementation for image-text similarity tasks."""

    # Dictionary of available model variants using structured configs
    _VARIANTS = {
        ModelVariant.BASE: ModelConfig(
            pretrained_model_name="line-corporation/clip-japanese-base",
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

        # Configuration parameters
        self.processor = None
        self.tokenizer = None
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
            model="CLIP_Japanese",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.MM_IMAGE_TEXT_SIM,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def _load_processor(self):
        """Load processor and tokenizer for the current variant.

        Returns:
            The loaded processor instance
        """
        pretrained_model_name = self._variant_config.pretrained_model_name
        self.processor = AutoImageProcessor.from_pretrained(
            pretrained_model_name, trust_remote_code=True
        )
        self.tokenizer = AutoTokenizer.from_pretrained(
            pretrained_model_name, trust_remote_code=True
        )

        return self.processor

    def load_model(self, *, dtype_override=None, **kwargs):
        """Load and return the CLIP Japanese model instance for this instance's variant.

        Args:
            dtype_override: Optional torch.dtype to override the model's default dtype.
                           If not provided, the model will use its default dtype (typically float32).

        Returns:
            torch.nn.Module: The CLIP Japanese model instance for image-text similarity.
        """
        pretrained_model_name = self._variant_config.pretrained_model_name

        model_kwargs = {"trust_remote_code": True}

        if dtype_override is not None:
            model_kwargs["torch_dtype"] = dtype_override
        model_kwargs |= kwargs

        model = AutoModel.from_pretrained(pretrained_model_name, **model_kwargs)
        model.eval()

        return model

    def load_inputs(self, dtype_override=None, batch_size=1):
        """Load and return sample inputs for the CLIP Japanese model.

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

        # Load image from HuggingFace dataset
        dataset = load_dataset("huggingface/cats-image")["test"]
        image = dataset[0]["image"]

        # Define text prompts for image-text similarity (Japanese)
        self.text_prompts = ["猫", "犬", "象"]

        # Process image
        image_inputs = self.processor(image, return_tensors="pt")

        # Tokenize text
        text_inputs = self.tokenizer(self.text_prompts, return_tensors="pt")

        # Replicate pixel_values to match the number of text prompts
        num_texts = len(self.text_prompts)
        image_inputs["pixel_values"] = image_inputs["pixel_values"].expand(
            num_texts, -1, -1, -1
        )

        # Combine inputs
        inputs = {**image_inputs, **text_inputs}

        # Replicate tensors for batch size
        for key in inputs:
            if torch.is_tensor(inputs[key]):
                inputs[key] = inputs[key].repeat_interleave(batch_size, dim=0)

        # Convert the input dtype to dtype_override if specified
        if dtype_override is not None:
            if "pixel_values" in inputs:
                inputs["pixel_values"] = inputs["pixel_values"].to(dtype_override)

        return inputs

    def post_process(self, outputs):
        """Post-process CLIP Japanese model outputs to extract similarity scores.

        Args:
            outputs: Raw model output

        Returns:
            dict: Post-processed similarity results with probabilities for each text prompt.
        """
        # Ensure text prompts are initialized
        if self.text_prompts is None:
            self.text_prompts = ["猫", "犬", "象"]

        # Extract image and text features from outputs
        image_features = outputs[0] if isinstance(outputs, tuple) else outputs
        text_features = outputs[1] if isinstance(outputs, tuple) else None

        if text_features is not None:
            # Compute similarity scores
            probs = (100.0 * image_features @ text_features.T).softmax(dim=-1)
            for i, text in enumerate(self.text_prompts):
                print(f"Probability of '{text}':", probs[0, i].item())

    def unpack_forward_output(self, fwd_output):
        """Unpack forward pass output to extract a differentiable tensor.

        Args:
            fwd_output: Output from the model's forward pass (tuple or object)

        Returns:
            torch.Tensor: Concatenated flattened outputs for backward pass
        """
        if isinstance(fwd_output, tuple):
            tensors = []
            for item in fwd_output:
                if isinstance(item, torch.Tensor):
                    tensors.append(item.flatten())
                elif hasattr(item, "last_hidden_state"):
                    tensors.append(item.last_hidden_state.flatten())
            if tensors:
                return torch.cat(tensors, dim=0)
        return fwd_output
