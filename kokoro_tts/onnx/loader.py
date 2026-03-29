# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Kokoro-82M v1.1 zh ONNX model loader implementation for text-to-speech tasks.
"""

from typing import Optional

import numpy as np
import onnx
import torch

from ...base import ForgeModel
from ...config import (
    Framework,
    ModelConfig,
    ModelGroup,
    ModelInfo,
    ModelSource,
    ModelTask,
    StrEnum,
)


class ModelVariant(StrEnum):
    """Available Kokoro ONNX model variants."""

    KOKORO_82M_V1_1_ZH = "82M-v1.1-zh"


class ModelLoader(ForgeModel):
    """Kokoro-82M v1.1 zh ONNX model loader for text-to-speech tasks."""

    _VARIANTS = {
        ModelVariant.KOKORO_82M_V1_1_ZH: ModelConfig(
            pretrained_model_name="suronek/Kokoro-82M-v1.1-zh-ONNX",
        ),
    }

    DEFAULT_VARIANT = ModelVariant.KOKORO_82M_V1_1_ZH

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT

        return ModelInfo(
            model="Kokoro",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.MM_TTS,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.ONNX,
        )

    def load_model(self, **kwargs):
        """Load and return the Kokoro ONNX model.

        Returns:
            onnx.ModelProto: The ONNX model instance.
        """
        from huggingface_hub import hf_hub_download

        local_path = hf_hub_download(
            repo_id=self._variant_config.pretrained_model_name,
            filename="model.onnx",
        )
        model = onnx.load(local_path)

        return model

    def load_inputs(self, **kwargs):
        """Load and return sample inputs for the Kokoro ONNX model.

        The model expects:
        - input_ids: Token IDs of shape [1, seq_len] (int64)
        - style: Style/voice embedding of shape [1, 256, style_dim] (float32)
        - speed: Speech speed scalar (float32)

        Returns:
            tuple: (input_ids, style, speed) sample input tensors.
        """
        seq_len = 20
        style_dim = 128

        input_ids = torch.randint(1, 178, (1, seq_len), dtype=torch.long)
        style = torch.randn(1, 256, style_dim, dtype=torch.float32)
        speed = torch.tensor([1.0], dtype=torch.float32)

        return input_ids, style, speed
