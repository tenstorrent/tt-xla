# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
LivePortrait appearance feature extractor model loader implementation.

LivePortrait is an efficient portrait animation framework that maps source images
to 3D appearance feature volumes for animation synthesis.
Source: https://huggingface.co/KlingTeam/LivePortrait
"""

from huggingface_hub import hf_hub_download
from torchvision import transforms
from datasets import load_dataset
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
    """Available LivePortrait model variants."""

    BASE = "Base"


class ModelLoader(ForgeModel):
    """LivePortrait appearance feature extractor model loader implementation."""

    _VARIANTS = {
        ModelVariant.BASE: ModelConfig(
            pretrained_model_name="KlingTeam/LivePortrait",
        ),
    }

    DEFAULT_VARIANT = ModelVariant.BASE

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self.transform = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT

        return ModelInfo(
            model="LivePortrait",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.CV_IMAGE_FE,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def _setup_transforms(self):
        self.transform = transforms.Compose(
            [
                transforms.Resize((256, 256)),
                transforms.ToTensor(),
            ]
        )
        return self.transform

    def load_model(self, *, dtype_override=None, **kwargs):
        from .src.model_utils import load_appearance_feature_extractor

        pretrained_model_name = self._variant_config.pretrained_model_name

        checkpoint_path = hf_hub_download(
            repo_id=pretrained_model_name,
            filename="liveportrait/base_models/appearance_feature_extractor.pth",
        )

        model = load_appearance_feature_extractor(checkpoint_path, device="cpu")

        if dtype_override is not None:
            model = model.to(dtype_override)

        return model

    def load_inputs(self, dtype_override=None, batch_size=1):
        if self.transform is None:
            self._setup_transforms()

        dataset = load_dataset("huggingface/cats-image")["test"]
        image = dataset[0]["image"].convert("RGB")

        inputs = self.transform(image).unsqueeze(0)

        if batch_size > 1:
            inputs = inputs.repeat(batch_size, 1, 1, 1)

        if dtype_override is not None:
            inputs = inputs.to(dtype_override)

        return inputs
