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

        model_kwargs = {"trust_remote_code": True}

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

        # KEEP model forward() takes (image_inputs, text_inputs) where
        # text_inputs is a dict of tokenizer outputs
        inputs = {
            "image_inputs": pixel_values,
            "text_inputs": dict(text_inputs),
        }

        # Replicate tensors for batch size
        inputs["image_inputs"] = inputs["image_inputs"].repeat_interleave(
            batch_size, dim=0
        )
        for key in inputs["text_inputs"]:
            if torch.is_tensor(inputs["text_inputs"][key]):
                inputs["text_inputs"][key] = inputs["text_inputs"][
                    key
                ].repeat_interleave(batch_size, dim=0)

        if dtype_override is not None:
            inputs["image_inputs"] = inputs["image_inputs"].to(dtype_override)

        return inputs

    def post_process(self, outputs):
        """Post-process KEEP model outputs to extract similarity scores.

        KEEP returns a dict with 'vision_features' and 'text_features' embeddings.
        Similarity is computed via dot product between normalized features.

        Args:
            outputs: Raw model output (dict with vision_features and text_features)
        """
        if self.text_prompts is None:
            self.text_prompts = ["a photo of a cat", "a photo of a dog"]

        vision_features = outputs["vision_features"]
        text_features = outputs["text_features"]

        # Compute cosine similarity (features are already L2-normalized by the model)
        similarity = vision_features @ text_features.T
        probs = similarity.softmax(dim=1)

        for i, text in enumerate(self.text_prompts):
            print(f"Probability of '{text}':", probs[0, i].item())

    def unpack_forward_output(self, fwd_output):
        """Unpack forward pass output to extract a differentiable tensor.

        KEEP returns a dict with 'vision_features' and 'text_features'.

        Args:
            fwd_output: Output from the model's forward pass (dict)

        Returns:
            torch.Tensor: Concatenated flattened outputs for backward pass
        """
        if isinstance(fwd_output, dict):
            tensors = []
            for value in fwd_output.values():
                if isinstance(value, torch.Tensor):
                    tensors.append(value.flatten())
            if tensors:
                return torch.cat(tensors, dim=0)
        return fwd_output
