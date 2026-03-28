# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
KEEP (Knowledge-Enhanced Evidence-based Pathology) model loader implementation for image-text similarity.
"""
import torch
from transformers import AutoModel, AutoTokenizer
from torchvision import transforms
from torchvision.transforms import InterpolationMode
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
    """Available KEEP model variants for pathology image-text similarity."""

    BASE = "Base"


class ModelLoader(ForgeModel):
    """KEEP model loader implementation for pathology image-text similarity tasks."""

    _VARIANTS = {
        ModelVariant.BASE: ModelConfig(
            pretrained_model_name="Astaxanthin/KEEP",
        ),
    }

    DEFAULT_VARIANT = ModelVariant.BASE

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self.processor = None
        self.tokenizer = None
        self.text_prompts = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        return ModelInfo(
            model="KEEP",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.MM_IMAGE_TEXT_SIM,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def _load_tokenizer(self):
        self.tokenizer = AutoTokenizer.from_pretrained(
            self._variant_config.pretrained_model_name,
            trust_remote_code=True,
        )
        return self.tokenizer

    def _get_image_transform(self):
        return transforms.Compose(
            [
                transforms.Resize(224, interpolation=InterpolationMode.BICUBIC),
                transforms.CenterCrop((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)
                ),
            ]
        )

    def load_model(self, *, dtype_override=None, **kwargs):
        """Load and return the KEEP model instance.

        Args:
            dtype_override: Optional torch.dtype to override the model's default dtype.

        Returns:
            torch.nn.Module: The KEEP model instance for pathology image-text similarity.
        """
        pretrained_model_name = self._variant_config.pretrained_model_name

        model_kwargs = {"trust_remote_code": True, "return_dict": False}

        if dtype_override is not None:
            model_kwargs["torch_dtype"] = dtype_override
        model_kwargs |= kwargs

        model = AutoModel.from_pretrained(pretrained_model_name, **model_kwargs)
        model.eval()

        return model

    def load_inputs(self, dtype_override=None, batch_size=1):
        """Load and return sample inputs for the KEEP model.

        Args:
            dtype_override: Optional torch.dtype to override the input dtype.
            batch_size: Optional batch size (default 1).

        Returns:
            dict: Input tensors containing pixel values and token inputs.
        """
        if self.tokenizer is None:
            self._load_tokenizer()

        # Load image from HuggingFace dataset
        dataset = load_dataset("huggingface/cats-image")["test"]
        image = dataset[0]["image"].convert("RGB")

        # Apply image preprocessing
        transform = self._get_image_transform()
        pixel_values = transform(image).unsqueeze(0)

        # Define text prompts for image-text similarity
        self.text_prompts = ["a photo of a cat", "a photo of a dog"]

        # Tokenize text inputs
        text_inputs = self.tokenizer(
            self.text_prompts,
            max_length=256,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        inputs = {
            "pixel_values": pixel_values,
            "input_ids": text_inputs["input_ids"],
            "attention_mask": text_inputs["attention_mask"],
        }

        # Replicate tensors for batch size
        for key in inputs:
            if torch.is_tensor(inputs[key]):
                inputs[key] = inputs[key].repeat_interleave(batch_size, dim=0)

        if dtype_override is not None:
            inputs["pixel_values"] = inputs["pixel_values"].to(dtype_override)

        return inputs

    def post_process(self, outputs):
        """Post-process KEEP model outputs to extract similarity scores.

        Args:
            outputs: Raw model output
        """
        if self.text_prompts is None:
            self.text_prompts = ["a photo of a cat", "a photo of a dog"]

        logits_per_image = outputs[0]
        probs = logits_per_image.softmax(dim=1)

        for i, text in enumerate(self.text_prompts):
            print(f"Probability of '{text}':", probs[0, i].item())

    def unpack_forward_output(self, fwd_output):
        """Unpack forward pass output to extract a differentiable tensor.

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
                    tensors.append(item.last_hidden_state.flatten())
            if tensors:
                return torch.cat(tensors, dim=0)
        return fwd_output
