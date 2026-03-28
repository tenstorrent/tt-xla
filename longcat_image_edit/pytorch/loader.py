# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
LongCat-Image-Edit model loader implementation for image editing
"""
import torch
from PIL import Image
from typing import Optional

from diffusers import LongCatImageEditPipeline

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
    """Available LongCat-Image-Edit model variants."""

    DEFAULT = "Default"


class ModelLoader(ForgeModel):
    """LongCat-Image-Edit model loader for bilingual image editing tasks."""

    _VARIANTS = {
        ModelVariant.DEFAULT: ModelConfig(
            pretrained_model_name="meituan-longcat/LongCat-Image-Edit",
        ),
    }

    DEFAULT_VARIANT = ModelVariant.DEFAULT

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self.pipe = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT

        return ModelInfo(
            model="LongCat-Image-Edit",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.CV_IMG_TO_IMG,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def _load_pipeline(self, dtype_override=None):
        pipe_kwargs = {}
        if dtype_override is not None:
            pipe_kwargs["torch_dtype"] = dtype_override

        self.pipe = LongCatImageEditPipeline.from_pretrained(
            self._variant_config.pretrained_model_name, **pipe_kwargs
        )
        self.pipe.enable_model_cpu_offload()

        return self.pipe

    def load_model(self, *, dtype_override=None, **kwargs):
        if self.pipe is None:
            self._load_pipeline(dtype_override=dtype_override)

        return self.pipe.transformer

    def load_inputs(self, dtype_override=None):
        if self.pipe is None:
            self._load_pipeline(dtype_override=dtype_override)

        dtype = dtype_override if dtype_override is not None else torch.bfloat16

        # Create a simple test image
        img = Image.new("RGB", (256, 256), color=(128, 128, 128))

        prompt = "Turn the background into a sunset"

        # Use the pipeline's encode methods to generate model inputs
        inputs = self.pipe(
            img,
            prompt,
            negative_prompt="",
            guidance_scale=4.5,
            num_inference_steps=1,
            num_images_per_prompt=1,
            generator=torch.Generator("cpu").manual_seed(43),
            output_type="latent",
        )

        return inputs
