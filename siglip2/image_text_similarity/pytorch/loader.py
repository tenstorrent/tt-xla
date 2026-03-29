# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
SigLIP2 model loader implementation for image-text similarity.
"""

import torch
import torch.nn.functional as F
from typing import Optional
from dataclasses import dataclass

from ....base import ForgeModel
from ....config import (
    ModelConfig,
    ModelInfo,
    ModelGroup,
    ModelTask,
    ModelSource,
    Framework,
    StrEnum,
)


@dataclass
class SigLIP2Config(ModelConfig):
    """Configuration specific to SigLIP2 models."""

    source: ModelSource = ModelSource.HUGGING_FACE


class ModelVariant(StrEnum):
    """Available SigLIP2 model variants for image-text similarity."""

    BASE_PATCH16_384 = "Base_Patch16_384"
    SO400M_PATCH14_384 = "So400m_Patch14_384"
    LARGE_PATCH16_256 = "Large_Patch16_256"

    # OpenCLIP variants
    VIT_SO400M_14_SIGLIP2 = "ViT_SO400M_14_SigLIP2"
    VIT_SO400M_16_SIGLIP2_256 = "ViT_SO400M_16_SigLIP2_256"


class ModelLoader(ForgeModel):
    """SigLIP2 model loader implementation for image-text similarity tasks."""

    # Dictionary of available model variants using structured configs
    _VARIANTS = {
        ModelVariant.SO400M_PATCH14_384: SigLIP2Config(
            pretrained_model_name="google/siglip2-so400m-patch14-384",
            source=ModelSource.HUGGING_FACE,
        ),
        ModelVariant.LARGE_PATCH16_256: SigLIP2Config(
            pretrained_model_name="google/siglip2-large-patch16-256",
            source=ModelSource.HUGGING_FACE,
        ),
        ModelVariant.VIT_SO400M_14_SIGLIP2: SigLIP2Config(
            pretrained_model_name="hf-hub:timm/ViT-SO400M-14-SigLIP2",
            source=ModelSource.TIMM,
        ),
        ModelVariant.VIT_SO400M_16_SIGLIP2_256: SigLIP2Config(
            pretrained_model_name="hf-hub:timm/ViT-SO400M-16-SigLIP2-256",
            source=ModelSource.TIMM,
        ),
    }

    # Default variant to use
    DEFAULT_VARIANT = ModelVariant.SO400M_PATCH14_384

    def __init__(self, variant: Optional[ModelVariant] = None):
        """Initialize ModelLoader with specified variant.

        Args:
            variant: Optional ModelVariant specifying which variant to use.
                     If None, DEFAULT_VARIANT is used.
        """
        super().__init__(variant)

        # Configuration parameters
        self.processor = None
        self.preprocess = None
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
        if variant is None:
            variant = cls.DEFAULT_VARIANT

        source = cls._VARIANTS[variant].source

        return ModelInfo(
            model="SigLIP2",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.MM_IMAGE_TEXT_SIM,
            source=source,
            framework=Framework.TORCH,
        )

    def _load_processor(self):
        """Load processor for the current variant (HuggingFace variants only).

        Returns:
            The loaded processor instance
        """
        from transformers import AutoProcessor

        self.processor = AutoProcessor.from_pretrained(
            self._variant_config.pretrained_model_name
        )

        return self.processor

    def load_model(self, *, dtype_override=None, **kwargs):
        """Load and return the SigLIP2 model instance for this instance's variant.

        Args:
            dtype_override: Optional torch.dtype to override the model's default dtype.
                           If not provided, the model will use its default dtype (typically float32).

        Returns:
            torch.nn.Module: The SigLIP2 model instance for image-text similarity.
        """
        source = self._variant_config.source

        if source == ModelSource.TIMM:
            from open_clip import create_model_from_pretrained, get_tokenizer

            pretrained_model_name = self._variant_config.pretrained_model_name

            model, self.preprocess = create_model_from_pretrained(pretrained_model_name)
            self.tokenizer = get_tokenizer(pretrained_model_name)

            if dtype_override is not None:
                model = model.to(dtype_override)

            model.eval()
            return model

        else:
            from transformers import AutoModel

            pretrained_model_name = self._variant_config.pretrained_model_name

            model_kwargs = {"return_dict": False}

            if dtype_override is not None:
                model_kwargs["torch_dtype"] = dtype_override
            model_kwargs |= kwargs

            model = AutoModel.from_pretrained(pretrained_model_name, **model_kwargs)
            model.eval()

            return model

    def load_inputs(self, dtype_override=None, batch_size=1):
        """Load and return sample inputs for the SigLIP2 model with this instance's variant settings.

        Args:
            dtype_override: Optional torch.dtype to override the model's default dtype.
                           If not provided, the model will use its default dtype (typically float32).
            batch_size: Optional batch size to override the default batch size of 1.

        Returns:
            dict: Input tensors that can be fed to the model.
        """
        from datasets import load_dataset

        dataset = load_dataset("huggingface/cats-image")["test"]
        image = dataset[0]["image"]

        self.text_prompts = ["a photo of 2 cats", "a photo of 2 dogs"]

        source = self._variant_config.source

        if source == ModelSource.TIMM:
            from open_clip import create_model_from_pretrained, get_tokenizer

            if self.preprocess is None or self.tokenizer is None:
                _, self.preprocess = create_model_from_pretrained(
                    self._variant_config.pretrained_model_name
                )
                self.tokenizer = get_tokenizer(
                    self._variant_config.pretrained_model_name
                )

            pixel_values = self.preprocess(image).unsqueeze(0)
            text_tokens = self.tokenizer(self.text_prompts)

            if batch_size > 1:
                pixel_values = pixel_values.repeat_interleave(batch_size, dim=0)
                text_tokens = text_tokens.repeat_interleave(batch_size, dim=0)

            if dtype_override is not None:
                pixel_values = pixel_values.to(dtype_override)

            return {"image": pixel_values, "text": text_tokens}

        else:
            if self.processor is None:
                self._load_processor()

            inputs = self.processor(
                text=self.text_prompts,
                images=image,
                return_tensors="pt",
                padding="max_length",
            )

            for key in inputs:
                if torch.is_tensor(inputs[key]):
                    inputs[key] = inputs[key].repeat_interleave(batch_size, dim=0)

            if dtype_override is not None:
                for key in inputs:
                    if (
                        torch.is_tensor(inputs[key])
                        and inputs[key].dtype == torch.float32
                    ):
                        inputs[key] = inputs[key].to(dtype_override)

            return inputs

    def post_process(self, outputs):
        """Post-process SigLIP2 model outputs to extract similarity scores.

        Args:
            outputs: Raw model output
        """
        source = self._variant_config.source

        if self.text_prompts is None:
            self.text_prompts = ["a photo of 2 cats", "a photo of 2 dogs"]

        if source == ModelSource.TIMM:
            image_features, text_features, logit_scale = outputs
            image_features = F.normalize(image_features, dim=-1)
            text_features = F.normalize(text_features, dim=-1)

            text_probs = torch.sigmoid(
                image_features @ text_features.T * logit_scale.exp()
            )

            for i, text in enumerate(self.text_prompts):
                print(f"Probability of '{text}':", text_probs[0, i].item())

    def unpack_forward_output(self, fwd_output):
        """Unpack forward pass output to extract a differentiable tensor.

        Args:
            fwd_output: Output from the model's forward pass (tuple of tensors)

        Returns:
            torch.Tensor: Concatenated flattened outputs for backward pass
        """
        if isinstance(fwd_output, tuple):
            tensors = []
            for item in fwd_output:
                if isinstance(item, torch.Tensor):
                    tensors.append(item.flatten())
            if tensors:
                return torch.cat(tensors, dim=0)
        return fwd_output
