# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
V-JEPA 2 model loader implementation for video feature extraction.
"""
import torch
from transformers import AutoModel, AutoVideoProcessor
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
    """Available V-JEPA 2 model variants."""

    VITG_FPC64_384 = "vitg_fpc64_384"


class ModelLoader(ForgeModel):
    """V-JEPA 2 model loader for video feature extraction."""

    _VARIANTS = {
        ModelVariant.VITG_FPC64_384: ModelConfig(
            pretrained_model_name="facebook/vjepa2-vitg-fpc64-384",
        ),
    }

    DEFAULT_VARIANT = ModelVariant.VITG_FPC64_384

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self.processor = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        return ModelInfo(
            model="VJEPA2",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.CV_VIDEO_CLS,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def _load_processor(self):
        self.processor = AutoVideoProcessor.from_pretrained(
            self._variant_config.pretrained_model_name
        )
        return self.processor

    def load_model(self, *, dtype_override=None, **kwargs):
        pretrained_model_name = self._variant_config.pretrained_model_name

        model_kwargs = {}
        if dtype_override is not None:
            model_kwargs["torch_dtype"] = dtype_override
        model_kwargs |= kwargs

        model = AutoModel.from_pretrained(pretrained_model_name, **model_kwargs)
        model.eval()

        return model

    def load_inputs(self, dtype_override=None, batch_size=1):
        if self.processor is None:
            self._load_processor()

        # Create synthetic video frames: 16 frames of 384x384 RGB
        num_frames = 16
        resolution = 384
        video = torch.randint(
            0, 255, (num_frames, 3, resolution, resolution), dtype=torch.uint8
        )

        inputs = self.processor(video, return_tensors="pt")

        # Replicate for batch size
        for key in inputs:
            if torch.is_tensor(inputs[key]):
                inputs[key] = inputs[key].repeat_interleave(batch_size, dim=0)

        if dtype_override is not None:
            if "pixel_values_videos" in inputs:
                inputs["pixel_values_videos"] = inputs["pixel_values_videos"].to(
                    dtype_override
                )

        return inputs
