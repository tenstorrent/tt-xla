# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
YOLOv5m license plate detection model loader implementation.

Loads the keremberke/yolov5m-license-plate model for detecting license plates
in images using the YOLOv5 medium architecture.

Available variants:
- YOLOV5M_LICENSE_PLATE: YOLOv5m fine-tuned for license plate detection
"""

from typing import Optional

import yolov5
from datasets import load_dataset

from ....base import ForgeModel
from ....config import (
    Framework,
    ModelConfig,
    ModelGroup,
    ModelInfo,
    ModelSource,
    ModelTask,
    StrEnum,
)
from ...pytorch.src.utils import data_postprocessing, data_preprocessing

HF_MODEL_REPO = "keremberke/yolov5m-license-plate"


class ModelVariant(StrEnum):
    """Available YOLOv5 license plate model variants."""

    YOLOV5M_LICENSE_PLATE = "yolov5m-license-plate"


class ModelLoader(ForgeModel):
    """YOLOv5m license plate detection model loader."""

    _VARIANTS = {
        ModelVariant.YOLOV5M_LICENSE_PLATE: ModelConfig(
            pretrained_model_name=HF_MODEL_REPO,
        ),
    }
    DEFAULT_VARIANT = ModelVariant.YOLOV5M_LICENSE_PLATE

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT
        return ModelInfo(
            model="YOLOv5_License_Plate",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.CV_OBJECT_DET,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def load_model(self, *, dtype_override=None, **kwargs):
        """Load the YOLOv5m license plate detection model.

        Returns:
            The YOLOv5 model instance configured for license plate detection.
        """
        model = yolov5.load(self._variant_config.pretrained_model_name)

        model.conf = 0.25
        model.iou = 0.45
        model.agnostic = False
        model.multi_label = False
        model.max_det = 1000

        if dtype_override is not None:
            model = model.to(dtype_override)

        return model

    def load_inputs(self, dtype_override=None, batch_size=1, input_size=640):
        """Load sample inputs for the license plate detection model.

        Returns:
            tuple: (ims, n, files, shape0, shape1, batch_tensor)
        """
        dataset = load_dataset("huggingface/cats-image")["test"]
        image_sample = dataset[0]["image"].convert("RGB")

        ims, n, files, shape0, shape1, img_tensor = data_preprocessing(
            image_sample, size=(input_size, input_size)
        )

        batch_tensor = img_tensor.repeat_interleave(batch_size, dim=0)

        if dtype_override is not None:
            batch_tensor = batch_tensor.to(dtype_override)

        return ims, n, files, shape0, shape1, batch_tensor

    def post_process(
        self, ims, pixel_values_shape, output, framework_model, n, shape0, shape1, files
    ):
        """Post-process model outputs to extract license plate detections."""
        results = data_postprocessing(
            ims,
            pixel_values_shape,
            output,
            framework_model,
            n,
            shape0,
            shape1,
            files,
        )

        print("Predictions:\n", results.pandas().xyxy)
