# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
YOLOv5n License Plate Detection model loader implementation
"""
from typing import Optional

from ...config import (
    ModelConfig,
    ModelInfo,
    ModelGroup,
    ModelTask,
    ModelSource,
    Framework,
    StrEnum,
)
from ...base import ForgeModel
from datasets import load_dataset
from .src.utils import data_preprocessing, data_postprocessing


class ModelVariant(StrEnum):
    """Available YOLOv5n License Plate model variants."""

    YOLOV5N_LICENSE_PLATE = "keremberke/yolov5n-license-plate"


class ModelLoader(ForgeModel):
    """YOLOv5n License Plate Detection model loader implementation."""

    _VARIANTS = {
        ModelVariant.YOLOV5N_LICENSE_PLATE: ModelConfig(
            pretrained_model_name="keremberke/yolov5n-license-plate",
        ),
    }

    DEFAULT_VARIANT = ModelVariant.YOLOV5N_LICENSE_PLATE

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT
        return ModelInfo(
            model="YOLOv5n-License-Plate",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.CV_OBJECT_DET,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def load_model(self, *, dtype_override=None, **kwargs):
        import yolov5

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
