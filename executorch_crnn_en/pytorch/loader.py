# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
ExecuTorch CRNN English model loader implementation for OCR text recognition tasks.

This loader wraps the EasyOCR CRNN recognizer model, which is the underlying
PyTorch architecture for the software-mansion/react-native-executorch-recognizer-crnn.en
HuggingFace model (distributed as ExecuTorch .pte files for mobile deployment).
"""
import torch
import torch.nn as nn
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


class CRNNWrapper(nn.Module):
    """Wrapper around the EasyOCR CRNN recognizer for a clean forward interface."""

    def __init__(self, recognizer):
        super().__init__()
        self.recognizer = recognizer

    def forward(self, x):
        return self.recognizer(x)


class ModelVariant(StrEnum):
    """Available ExecuTorch CRNN English model variants."""

    BASE = "Base"


class ModelLoader(ForgeModel):
    """ExecuTorch CRNN English model loader for OCR text recognition."""

    _VARIANTS = {
        ModelVariant.BASE: ModelConfig(
            pretrained_model_name="software-mansion/react-native-executorch-recognizer-crnn.en",
        ),
    }

    DEFAULT_VARIANT = ModelVariant.BASE

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        return ModelInfo(
            model="executorch_crnn_en",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.CV_IMAGE_TO_TEXT,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def load_model(self, *, dtype_override=None, **kwargs):
        import easyocr

        reader = easyocr.Reader(["en"], gpu=False)
        model = reader.recognizer
        model.eval()

        if dtype_override is not None:
            model = model.to(dtype_override)

        return CRNNWrapper(model)

    def load_inputs(self, dtype_override=None, batch_size=1):
        # CRNN recognizer expects grayscale images with height 64
        image = Image.new("L", (200, 64), color=128)
        tensor = (
            torch.from_numpy(np.array(image)).float().unsqueeze(0).unsqueeze(0) / 255.0
        )

        if dtype_override is not None:
            tensor = tensor.to(dtype_override)

        if batch_size > 1:
            tensor = tensor.repeat(batch_size, 1, 1, 1)

        return tensor
