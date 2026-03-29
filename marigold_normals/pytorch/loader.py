# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Marigold Normals model loader implementation for monocular surface normal estimation.
"""
import torch
from diffusers import MarigoldNormalsPipeline
from datasets import load_dataset
from typing import Optional

from ...config import (
    ModelConfig,
    ModelInfo,
    ModelGroup,
    ModelTask,
    ModelSource,
    Framework,
    StrEnum,
)
from ...base import ForgeModel


class ModelVariant(StrEnum):
    """Available Marigold Normals model variants."""

    V0_1 = "v0.1"


class ModelLoader(ForgeModel):
    """Marigold Normals model loader implementation."""

    _VARIANTS = {
        ModelVariant.V0_1: ModelConfig(
            pretrained_model_name="prs-eth/marigold-normals-v0-1",
        ),
    }

    DEFAULT_VARIANT = ModelVariant.V0_1

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self.pipeline = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT

        return ModelInfo(
            model="MarigoldNormals",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.CV_DEPTH_EST,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def _load_pipeline(self, dtype_override=None):
        pretrained_model_name = self._variant_config.pretrained_model_name
        pipe = MarigoldNormalsPipeline.from_pretrained(
            pretrained_model_name, torch_dtype=dtype_override or torch.float32
        )
        pipe.to("cpu")

        for module in [pipe.unet, pipe.vae, pipe.text_encoder]:
            module.eval()
            for param in module.parameters():
                param.requires_grad = False

        self.pipeline = pipe
        return pipe

    def load_model(self, *, dtype_override=None, **kwargs):
        if self.pipeline is None:
            self._load_pipeline(dtype_override=dtype_override)

        return self.pipeline

    def load_inputs(self, dtype_override=None, batch_size=1):
        if self.pipeline is None:
            self._load_pipeline(dtype_override=dtype_override)

        pipe = self.pipeline

        dataset = load_dataset("huggingface/cats-image")["test"]
        image = dataset[0]["image"]

        inputs = {"image": image, "num_inference_steps": 10, "batch_size": batch_size}

        return inputs
