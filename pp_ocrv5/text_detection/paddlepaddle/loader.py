# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
PP-OCRv5 Server Detection PaddlePaddle model loader implementation.
"""

from typing import Optional

import paddle
from PIL import Image

from ....config import (
    ModelConfig,
    ModelInfo,
    ModelGroup,
    ModelTask,
    ModelSource,
    Framework,
    StrEnum,
)
from ....base import ForgeModel
from ....tools.utils import get_file


class ModelVariant(StrEnum):
    """Available PP-OCRv5 text detection model variants (Paddle)."""

    SERVER_DET = "PP-OCRv5_server_det"


class ModelLoader(ForgeModel):
    """PP-OCRv5 Server Detection PaddlePaddle model loader implementation."""

    _VARIANTS = {
        ModelVariant.SERVER_DET: ModelConfig(
            pretrained_model_name="PP-OCRv5_server_det",
        ),
    }

    DEFAULT_VARIANT = ModelVariant.SERVER_DET

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        """Get model information for dashboard and metrics reporting."""
        if variant is None:
            variant = cls.DEFAULT_VARIANT

        return ModelInfo(
            model="PP-OCRv5",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.CV_OBJECT_DET,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.PADDLE,
        )

    def load_model(self, *, dtype_override=None, **kwargs):
        """Load pretrained PP-OCRv5 server detection model (Paddle)."""
        import os

        from paddlex.inference import create_predictor

        predictor = create_predictor(model_name="PP-OCRv5_server_det")
        model = paddle.jit.load(os.path.join(str(predictor.model_dir), "inference"))
        model.eval()
        return model

    def load_inputs(self, dtype_override=None, batch_size: int = 1):
        """Prepare sample input for PP-OCRv5 detection model (Paddle)."""
        import numpy as np

        image_file = get_file(
            "https://github.com/pytorch/hub/raw/master/images/dog.jpg"
        )
        image = Image.open(str(image_file)).convert("RGB")

        # Follow PP-OCRv5 preprocessing: resize longest side to 960, normalize
        image = np.array(image).astype("float32")
        h, w = image.shape[:2]
        ratio = 960.0 / max(h, w)
        new_h = int(h * ratio)
        new_w = int(w * ratio)
        # Round to multiple of 32 for the detector
        new_h = max(32, (new_h + 31) // 32 * 32)
        new_w = max(32, (new_w + 31) // 32 * 32)

        image = Image.fromarray(image.astype("uint8")).resize(
            (new_w, new_h), Image.BILINEAR
        )
        image = np.array(image).astype("float32")

        # Normalize
        mean = np.array([0.485, 0.456, 0.406], dtype="float32")
        std = np.array([0.229, 0.224, 0.225], dtype="float32")
        image = (image / 255.0 - mean) / std

        # HWC -> CHW
        image = image.transpose((2, 0, 1))

        inputs = paddle.to_tensor(image).unsqueeze(0)

        if batch_size and batch_size > 1:
            inputs = paddle.tile(inputs, repeat_times=[batch_size, 1, 1, 1])

        return [inputs]
