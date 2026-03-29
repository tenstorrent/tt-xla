# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Japanese CLOOB model loader implementation for image-text similarity.

Uses the japanese_clip package to load the rinna/japanese-cloob-vit-b-16 model,
which performs contrastive image-text matching with Japanese text.
"""
import torch
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
    """Available Japanese CLOOB model variants."""

    VIT_B_16 = "ViT_B_16"


class ModelLoader(ForgeModel):
    """Japanese CLOOB model loader implementation for image-text similarity tasks."""

    _VARIANTS = {
        ModelVariant.VIT_B_16: ModelConfig(
            pretrained_model_name="rinna/japanese-cloob-vit-b-16",
        ),
    }

    DEFAULT_VARIANT = ModelVariant.VIT_B_16

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self.preprocess = None
        self.tokenizer = None
        self.text_prompts = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        return ModelInfo(
            model="Japanese_CLOOB",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.MM_IMAGE_TEXT_SIM,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def load_model(self, *, dtype_override=None, **kwargs):
        """Load and return the Japanese CLOOB model instance.

        Args:
            dtype_override: Optional torch.dtype to override the model's default dtype.

        Returns:
            torch.nn.Module: The Japanese CLOOB model instance.
        """
        import japanese_clip as ja_clip

        model, self.preprocess = ja_clip.load(
            self._variant_config.pretrained_model_name, device="cpu"
        )
        self.tokenizer = ja_clip.load_tokenizer()

        if dtype_override is not None:
            model = model.to(dtype_override)

        model.eval()
        return model

    def load_inputs(self, dtype_override=None, batch_size=1):
        """Load and return sample inputs for the Japanese CLOOB model.

        Args:
            dtype_override: Optional torch.dtype to override the model's default dtype.
            batch_size: Optional batch size to override the default batch size of 1.

        Returns:
            dict: Input tensors that can be fed to the model.
        """
        import japanese_clip as ja_clip

        if self.preprocess is None or self.tokenizer is None:
            self.load_model()

        # Load image from HuggingFace dataset
        dataset = load_dataset("huggingface/cats-image")["test"]
        image = dataset[0]["image"]

        # Preprocess the image
        pixel_values = self.preprocess(image).unsqueeze(0)

        # Define Japanese text prompts (cat, dog, elephant)
        self.text_prompts = ["猫", "犬", "象"]

        # Tokenize text
        encodings = ja_clip.tokenize(
            texts=self.text_prompts,
            max_seq_len=77,
            device="cpu",
            tokenizer=self.tokenizer,
        )

        # Replicate tensors for batch size
        pixel_values = pixel_values.repeat_interleave(batch_size, dim=0)
        for key in encodings:
            if torch.is_tensor(encodings[key]):
                encodings[key] = encodings[key].repeat_interleave(batch_size, dim=0)

        if dtype_override is not None:
            pixel_values = pixel_values.to(dtype_override)

        return {"image": pixel_values, **encodings}

    def post_process(self, outputs):
        """Post-process model outputs to extract similarity scores.

        Args:
            outputs: Raw model output (image_features, text_features)
        """
        if self.text_prompts is None:
            self.text_prompts = ["猫", "犬", "象"]

        image_features, text_features = outputs[0], outputs[1]

        # Normalize features
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        # Compute similarity
        similarity = (image_features @ text_features.T).softmax(dim=-1)

        for i, text in enumerate(self.text_prompts):
            print(f"Probability of '{text}':", similarity[0, i].item())
