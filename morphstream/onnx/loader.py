# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
MorphStream face swap model loader implementation
"""

import torch
import onnx
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
    """Available MorphStream model variants."""

    INSWAPPER_128_FP16 = "inswapper_128_fp16"


class ModelLoader(ForgeModel):
    """MorphStream face swap model loader implementation."""

    _VARIANTS = {
        ModelVariant.INSWAPPER_128_FP16: ModelConfig(
            pretrained_model_name="latark/MorphStream",
        ),
    }

    DEFAULT_VARIANT = ModelVariant.INSWAPPER_128_FP16

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT

        return ModelInfo(
            model="MorphStream",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.CV_IMG_TO_IMG,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.ONNX,
        )

    def load_model(self, **kwargs):
        from huggingface_hub import hf_hub_download

        pretrained = self._variant_config.pretrained_model_name
        model_path = hf_hub_download(pretrained, "inswapper_128_fp16.onnx")

        model = onnx.load(model_path)
        return model

    def load_inputs(self, **kwargs):
        # InSwapper expects a 128x128 face crop (3 channels) and a 512-dim face embedding
        face_input = torch.randn(1, 3, 128, 128)
        face_embedding = torch.randn(1, 512)

        return {"target": face_input, "source": face_embedding}
