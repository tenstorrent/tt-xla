# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
EVF-SAM2 (Early Vision-Language Fusion SAM2) loader implementation
for text-prompted image segmentation.
"""

import torch
from typing import Optional
from PIL import Image
from loguru import logger
from transformers import AutoModel, AutoTokenizer
from datasets import load_dataset

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
    """Available EVF-SAM2 model variants."""

    MULTITASK = "Multitask"


class ModelLoader(ForgeModel):
    """EVF-SAM2 model loader implementation for text-prompted image segmentation."""

    _VARIANTS = {
        ModelVariant.MULTITASK: ModelConfig(
            pretrained_model_name="YxZhang/evf-sam2-multitask",
        ),
    }

    DEFAULT_VARIANT = ModelVariant.MULTITASK

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self.tokenizer = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT
        return ModelInfo(
            model="EVF-SAM2",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.CV_IMAGE_SEG,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def _load_tokenizer(self):
        self.tokenizer = AutoTokenizer.from_pretrained(
            self._variant_config.pretrained_model_name,
            padding_side="right",
            use_fast=False,
            trust_remote_code=True,
        )
        return self.tokenizer

    def load_model(self, *, dtype_override=None, **kwargs):
        pretrained_model_name = self._variant_config.pretrained_model_name

        if self.tokenizer is None:
            self._load_tokenizer()

        model_kwargs = {"trust_remote_code": True}
        if dtype_override is not None:
            model_kwargs["torch_dtype"] = dtype_override
        model_kwargs |= kwargs

        model = AutoModel.from_pretrained(pretrained_model_name, **model_kwargs)
        model.eval()

        return model

    def load_inputs(self, dtype_override=None, batch_size=1):
        if self.tokenizer is None:
            self._load_tokenizer()

        try:
            dataset = load_dataset("huggingface/cats-image")["test"]
            raw_image = dataset[0]["image"].convert("RGB")
        except Exception as e:
            logger.warning(
                f"Failed to load image from dataset. Using random fallback tensor. Reason: {e}"
            )
            raw_image = Image.fromarray(
                (torch.rand(3, 224, 224) * 255).byte().permute(1, 2, 0).numpy()
            )

        # Tokenize the text prompt
        input_ids = self.tokenizer("a cat", return_tensors="pt", padding=True)[
            "input_ids"
        ]

        # Preprocess image for BEiT-3 (224x224, normalized to [-1, 1])
        image_beit = raw_image.resize((224, 224))
        image_beit = torch.tensor(
            list(image_beit.getdata()), dtype=torch.float32
        ).reshape(1, 224, 224, 3)
        image_beit = image_beit.permute(0, 3, 1, 2) / 127.5 - 1.0

        # Preprocess image for SAM2 (1024x1024, ImageNet normalization)
        image_sam = raw_image.resize((1024, 1024))
        image_sam = torch.tensor(
            list(image_sam.getdata()), dtype=torch.float32
        ).reshape(1, 1024, 1024, 3)
        image_sam = image_sam.permute(0, 3, 1, 2) / 255.0
        mean = torch.tensor([0.485, 0.456, 0.406]).reshape(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).reshape(1, 3, 1, 1)
        image_sam = (image_sam - mean) / std

        if dtype_override is not None:
            image_beit = image_beit.to(dtype_override)
            image_sam = image_sam.to(dtype_override)

        if batch_size > 1:
            input_ids = input_ids.repeat_interleave(batch_size, dim=0)
            image_beit = image_beit.repeat_interleave(batch_size, dim=0)
            image_sam = image_sam.repeat_interleave(batch_size, dim=0)

        return {
            "images_evf": image_beit,
            "images_sam": image_sam,
            "input_ids": input_ids,
        }
