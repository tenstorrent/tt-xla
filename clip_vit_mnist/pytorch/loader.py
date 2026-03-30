# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
CLIP ViT Base Patch32 MNIST vision encoder model loader for image feature extraction.
"""
import torch
from transformers import CLIPVisionModel, CLIPImageProcessor
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
    """Available CLIP ViT Base Patch32 MNIST model variants."""

    BASE_PATCH32_MNIST = "Base_Patch32_MNIST"


class ModelLoader(ForgeModel):
    """CLIP ViT Base Patch32 MNIST vision encoder model loader for image feature extraction."""

    _VARIANTS = {
        ModelVariant.BASE_PATCH32_MNIST: ModelConfig(
            pretrained_model_name="tanganke/clip-vit-base-patch32_mnist",
        ),
    }

    DEFAULT_VARIANT = ModelVariant.BASE_PATCH32_MNIST

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self.processor = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        return ModelInfo(
            model="CLIP_VIT_MNIST",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.CV_IMAGE_FE,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def load_model(self, *, dtype_override=None, **kwargs):
        """Load and return the CLIPVisionModel instance.

        Args:
            dtype_override: Optional torch.dtype to override the model's default dtype.

        Returns:
            torch.nn.Module: The CLIP vision model instance.
        """
        pretrained_model_name = self._variant_config.pretrained_model_name

        model_kwargs = {}
        if dtype_override is not None:
            model_kwargs["torch_dtype"] = dtype_override
        model_kwargs |= kwargs

        model = CLIPVisionModel.from_pretrained(pretrained_model_name, **model_kwargs)
        model.eval()

        return model

    def load_inputs(self, dtype_override=None, batch_size=1):
        """Load and return sample inputs for the CLIP vision model.

        Args:
            dtype_override: Optional torch.dtype to override the input dtype.
            batch_size: Optional batch size (default 1).

        Returns:
            dict: Input tensors containing pixel values.
        """
        if self.processor is None:
            self.processor = CLIPImageProcessor.from_pretrained(
                self._variant_config.pretrained_model_name
            )

        dataset = load_dataset("huggingface/cats-image")["test"]
        image = dataset[0]["image"]

        inputs = self.processor(images=image, return_tensors="pt")

        if batch_size > 1:
            for key in inputs:
                if torch.is_tensor(inputs[key]):
                    inputs[key] = inputs[key].repeat_interleave(batch_size, dim=0)

        if dtype_override is not None:
            inputs["pixel_values"] = inputs["pixel_values"].to(dtype_override)

        return inputs

    def post_process(self, outputs):
        """Post-process model outputs.

        Args:
            outputs: Raw model output containing last_hidden_state and pooler_output.
        """
        if hasattr(outputs, "pooler_output"):
            pooler_output = outputs.pooler_output
        else:
            pooler_output = outputs[1]

        print(f"Pooler output shape: {pooler_output.shape}")
        print(f"Pooler output (first 5 values): {pooler_output[0, :5]}")

    def unpack_forward_output(self, fwd_output):
        """Unpack forward pass output to extract a differentiable tensor.

        Args:
            fwd_output: Output from the model's forward pass.

        Returns:
            torch.Tensor: Concatenated flattened outputs for backward pass.
        """
        if isinstance(fwd_output, tuple):
            tensors = []
            for item in fwd_output:
                if isinstance(item, torch.Tensor):
                    tensors.append(item.flatten())
            if tensors:
                return torch.cat(tensors, dim=0)
        if hasattr(fwd_output, "last_hidden_state"):
            return fwd_output.last_hidden_state.flatten()
        return fwd_output
