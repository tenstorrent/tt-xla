# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
PP-OCRv5 Server Recognition PaddlePaddle model loader implementation.
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

    SERVER_REC = "PP-OCRv5_server_rec"


class ModelLoader(ForgeModel):
    """PP-OCRv5 Server Recognition PaddlePaddle model loader implementation."""

    _VARIANTS = {
        ModelVariant.SERVER_REC: ModelConfig(
            pretrained_model_name="PP-OCRv5_server_rec",
        ),
    }

    DEFAULT_VARIANT = ModelVariant.SERVER_REC

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        """Get model information for dashboard and metrics reporting."""
        if variant is None:
            variant = cls.DEFAULT_VARIANT

        return ModelInfo(
            model="PP-OCRv5",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.CV_IMAGE_TO_TEXT,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.PADDLE,
        )

    def load_model(self, *, dtype_override=None, **kwargs):
        """Load pretrained PP-OCRv5 server recognition model (Paddle)."""
        import os

        from paddlex.inference import create_predictor

        predictor = create_predictor(model_name="PP-OCRv5_server_rec")
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

        # PP-OCRv5 recognition expects input shape [batch, 3, 48, 320]
        image = image.resize((320, 48), Image.BILINEAR)
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
