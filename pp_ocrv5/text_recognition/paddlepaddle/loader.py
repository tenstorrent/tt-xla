# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
PP-OCRv5 Latin Mobile Recognition PaddlePaddle model loader implementation.
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
    """Available PP-OCRv5 text recognition model variants (Paddle)."""

    LATIN_MOBILE_REC = "latin_PP-OCRv5_mobile_rec"


class ModelLoader(ForgeModel):
    """PP-OCRv5 Latin Mobile Recognition PaddlePaddle model loader implementation."""

    _VARIANTS = {
        ModelVariant.LATIN_MOBILE_REC: ModelConfig(
            pretrained_model_name="latin_PP-OCRv5_mobile_rec",
        ),
    }

    DEFAULT_VARIANT = ModelVariant.LATIN_MOBILE_REC

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        """Get model information for dashboard and metrics reporting."""
        if variant is None:
            variant = cls.DEFAULT_VARIANT

        return ModelInfo(
            model="PP-OCRv5",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.MM_DOC_OCR,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.PADDLE,
        )

    def load_model(self, *, dtype_override=None, **kwargs):
        """Load pretrained PP-OCRv5 latin mobile recognition model (Paddle)."""
        import os

        from paddlex.inference import create_predictor

        predictor = create_predictor(model_name="latin_PP-OCRv5_mobile_rec")
        model = paddle.jit.load(os.path.join(str(predictor.model_dir), "inference"))
        model.eval()
        return model

    def load_inputs(self, dtype_override=None, batch_size: int = 1):
        """Prepare sample input for PP-OCRv5 recognition model (Paddle)."""
        import numpy as np

        image_file = get_file(
            "https://github.com/pytorch/hub/raw/master/images/dog.jpg"
        )
        image = Image.open(str(image_file)).convert("RGB")

        # Resize to fixed height 48 with aspect ratio preserved for recognition
        image = np.array(image).astype("float32")
        h, w = image.shape[:2]
        target_h = 48
        ratio = target_h / h
        new_w = max(1, int(w * ratio))
        # Round width to multiple of 4
        new_w = max(4, (new_w + 3) // 4 * 4)

        image = Image.fromarray(image.astype("uint8")).resize(
            (new_w, target_h), Image.BILINEAR
        )
        image = np.array(image).astype("float32")

        # Normalize
        mean = np.array([0.5, 0.5, 0.5], dtype="float32")
        std = np.array([0.5, 0.5, 0.5], dtype="float32")
        image = (image / 255.0 - mean) / std

        # HWC -> CHW
        image = image.transpose((2, 0, 1))

        inputs = paddle.to_tensor(image).unsqueeze(0)

        if batch_size and batch_size > 1:
            inputs = paddle.tile(inputs, repeat_times=[batch_size, 1, 1, 1])

        return [inputs]
