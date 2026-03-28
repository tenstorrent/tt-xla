# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
CoCa (Contrastive Captioners) model loader implementation for image-text similarity using OpenCLIP.
"""
import torch
import torch.nn.functional as F
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
    """Available CoCa model variants."""

    VIT_L_14_LAION2B = "ViT_L_14_laion2B"


# Mapping from variant to OpenCLIP tokenizer name
_TOKENIZER_NAME = {
    ModelVariant.VIT_L_14_LAION2B: "coca_ViT-L-14",
}


class ModelLoader(ForgeModel):
    """CoCa model loader using OpenCLIP for image-text similarity tasks."""

    _VARIANTS = {
        ModelVariant.VIT_L_14_LAION2B: ModelConfig(
            pretrained_model_name="hf-hub:laion/CoCa-ViT-L-14-laion2B-s13B-b90k",
        ),
    }

    DEFAULT_VARIANT = ModelVariant.VIT_L_14_LAION2B

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self.preprocess = None
        self.tokenizer = None
        self.text_prompts = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        return ModelInfo(
            model="COCA",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.MM_IMAGE_TEXT_SIM,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def load_model(self, *, dtype_override=None, **kwargs):
        """Load and return the CoCa model instance.

        Args:
            dtype_override: Optional torch.dtype to override the model's default dtype.

        Returns:
            torch.nn.Module: The CoCa model instance.
        """
        from open_clip import create_model_from_pretrained, get_tokenizer

        pretrained_model_name = self._variant_config.pretrained_model_name

        model, self.preprocess = create_model_from_pretrained(pretrained_model_name)
        self.tokenizer = get_tokenizer(_TOKENIZER_NAME[self._variant])

        if dtype_override is not None:
            model = model.to(dtype_override)

        model.eval()
        return model

    def load_inputs(self, dtype_override=None, batch_size=1):
        """Load and return sample inputs for the CoCa model.

        Args:
            dtype_override: Optional torch.dtype to override the input dtype.
            batch_size: Optional batch size (default 1).

        Returns:
            dict: Input tensors containing image and text tokens.
        """
        from open_clip import create_model_from_pretrained, get_tokenizer

        if self.preprocess is None or self.tokenizer is None:
            _, self.preprocess = create_model_from_pretrained(
                self._variant_config.pretrained_model_name
            )
            self.tokenizer = get_tokenizer(_TOKENIZER_NAME[self._variant])

        # Load image from HuggingFace dataset
        from datasets import load_dataset

        dataset = load_dataset("huggingface/cats-image")["test"]
        image = dataset[0]["image"]

        self.text_prompts = ["a photo of a cat", "a photo of a dog"]

        # Preprocess image
        pixel_values = self.preprocess(image).unsqueeze(0)

        # Tokenize text
        text_tokens = self.tokenizer(self.text_prompts)

        # Replicate for batch size
        if batch_size > 1:
            pixel_values = pixel_values.repeat_interleave(batch_size, dim=0)
            text_tokens = text_tokens.repeat_interleave(batch_size, dim=0)

        if dtype_override is not None:
            pixel_values = pixel_values.to(dtype_override)

        return {"image": pixel_values, "text": text_tokens}

    def post_process(self, outputs):
        """Post-process CoCa model outputs to extract similarity scores.

        Args:
            outputs: Raw model output (image_features, text_features, logit_scale)
        """
        if self.text_prompts is None:
            self.text_prompts = ["a photo of a cat", "a photo of a dog"]

        image_features, text_features, logit_scale = outputs
        image_features = F.normalize(image_features, dim=-1)
        text_features = F.normalize(text_features, dim=-1)

        text_probs = torch.sigmoid(image_features @ text_features.T * logit_scale.exp())

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
