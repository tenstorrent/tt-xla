# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
OnnxTR PARSeq Multilingual model loader implementation for text recognition.

This model is a PARSeq (Permuted Autoregressive Sequence) text recognition model
exported to ONNX format, supporting 12 languages with Latin-based scripts.
"""
import torch
import onnx
import numpy as np
from PIL import Image
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
    """Available OnnxTR PARSeq Multilingual model variants."""

    PARSEQ_MULTILINGUAL_V1 = "parseq-multilingual-v1"


class ModelLoader(ForgeModel):
    """OnnxTR PARSeq Multilingual model loader for text recognition."""

    _VARIANTS = {
        ModelVariant.PARSEQ_MULTILINGUAL_V1: ModelConfig(
            pretrained_model_name="Felix92/onnxtr-parseq-multilingual-v1",
        ),
    }

    DEFAULT_VARIANT = ModelVariant.PARSEQ_MULTILINGUAL_V1

    # Model input normalization parameters from config.json
    INPUT_HEIGHT = 32
    INPUT_WIDTH = 128
    MEAN = [0.694, 0.695, 0.693]
    STD = [0.299, 0.296, 0.301]

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        return ModelInfo(
            model="onnxtr_parseq_multilingual",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.CV_IMAGE_TO_TEXT,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.ONNX,
        )

    def load_model(self, *, dtype_override=None, **kwargs):
        """Load the ONNX model from HuggingFace Hub."""
        from huggingface_hub import hf_hub_download

        pretrained = self._variant_config.pretrained_model_name
        model_path = hf_hub_download(pretrained, "model.onnx")
        model = onnx.load(model_path)

        return model

    def load_inputs(self, *, dtype_override=None, batch_size=1):
        """Prepare sample input for the PARSeq text recognition model.

        Creates a synthetic text-like image normalized with the model's
        expected mean and std values. Input shape: [batch, 3, 32, 128].
        """
        image = Image.new(
            "RGB", (self.INPUT_WIDTH, self.INPUT_HEIGHT), color=(200, 200, 200)
        )

        img_array = np.array(image, dtype=np.float32) / 255.0
        # Normalize with model-specific mean and std
        mean = np.array(self.MEAN, dtype=np.float32)
        std = np.array(self.STD, dtype=np.float32)
        img_array = (img_array - mean) / std
        # HWC -> CHW
        img_array = img_array.transpose(2, 0, 1)

        inputs = torch.from_numpy(img_array).unsqueeze(0)

        if dtype_override is not None:
            inputs = inputs.to(dtype_override)

        inputs = inputs.repeat_interleave(batch_size, dim=0)

        return inputs
