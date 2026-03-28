# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
RT-DETR-L wired table cell detection PaddlePaddle model loader implementation.
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
    """Available RT-DETR table cell detection model variants (Paddle)."""

    L_WIRED = "RT-DETR-L_wired_table_cell_det"


class ModelLoader(ForgeModel):
    """RT-DETR-L wired table cell detection PaddlePaddle model loader."""

    _VARIANTS = {
        ModelVariant.L_WIRED: ModelConfig(
            pretrained_model_name="RT-DETR-L_wired_table_cell_det",
        ),
    }

    DEFAULT_VARIANT = ModelVariant.L_WIRED

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        """Get model information for dashboard and metrics reporting."""
        if variant is None:
            variant = cls.DEFAULT_VARIANT

        return ModelInfo(
            model="RT-DETR",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.CV_OBJECT_DET,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.PADDLE,
        )

    def load_model(self, *, dtype_override=None, **kwargs):
        """Load pretrained RT-DETR-L wired table cell detection model (Paddle)."""
        import os

        from paddlex.inference import create_predictor

        predictor = create_predictor(
            model_name="RT-DETR-L_wired_table_cell_det",
        )
        model = paddle.jit.load(os.path.join(str(predictor.model_dir), "inference"))
        model.eval()
        return model

    def load_inputs(self, dtype_override=None, batch_size: int = 1):
        """Prepare sample input for RT-DETR-L table cell detection model (Paddle)."""
        import numpy as np

        image_file = get_file(
            "https://github.com/pytorch/hub/raw/master/images/dog.jpg"
        )
        image = Image.open(str(image_file)).convert("RGB")

        # RT-DETR preprocessing: resize to 640x640, normalize
        image = image.resize((640, 640), Image.BILINEAR)
        image = np.array(image).astype("float32")

        # Normalize with ImageNet statistics
        mean = np.array([0.485, 0.456, 0.406], dtype="float32")
        std = np.array([0.229, 0.224, 0.225], dtype="float32")
        image = (image / 255.0 - mean) / std

        # HWC -> CHW
        image = image.transpose((2, 0, 1))

        inputs = paddle.to_tensor(image).unsqueeze(0)

        if batch_size and batch_size > 1:
            inputs = paddle.tile(inputs, repeat_times=[batch_size, 1, 1, 1])

        return [inputs]
