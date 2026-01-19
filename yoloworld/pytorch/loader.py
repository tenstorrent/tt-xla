# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0
"""
YOLO-World model loader implementation
"""

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
from typing import Optional
from ...tools.utils import get_file


class ModelVariant(StrEnum):
    """Available YOLO WORLD model variants."""

    SMALL_1 = "small_640"
    SMALL_2 = "small_1280"
    MEDIUM_1 = "medium_640"
    MEDIUM_2 = "medium_1280"
    LARGE_1 = "large_640"
    LARGE_2 = "large_1280"
    XLARGE_1 = "xlarge_640"


class ModelLoader(ForgeModel):
    """YOLO WORLD model loader."""

    _VARIANTS = {
        ModelVariant.SMALL_1: ModelConfig(
            pretrained_model_name="small_640",
        ),
        ModelVariant.SMALL_2: ModelConfig(
            pretrained_model_name="small_1280",
        ),
        ModelVariant.MEDIUM_1: ModelConfig(
            pretrained_model_name="medium_640",
        ),
        ModelVariant.MEDIUM_2: ModelConfig(
            pretrained_model_name="medium_1280",
        ),
        ModelVariant.LARGE_1: ModelConfig(
            pretrained_model_name="large_640",
        ),
        ModelVariant.LARGE_2: ModelConfig(
            pretrained_model_name="large_1280",
        ),
        ModelVariant.XLARGE_1: ModelConfig(
            pretrained_model_name="xlarge_640",
        ),
    }

    DEFAULT_VARIANT = ModelVariant.SMALL_1
    DEFAULT_TOPK = 100
    DEFAULT_THRESHOLD = 0.005
    DEFAULT_TEXTS = "person,bus"

    def __init__(
        self,
        variant: Optional[ModelVariant] = None,
        topk: Optional[int] = None,
        threshold: Optional[float] = None,
        texts: Optional[str] = None,
    ):
        super().__init__()
        self.variant = variant or self.DEFAULT_VARIANT
        self.topk = topk or self.DEFAULT_TOPK
        self.threshold = threshold or self.DEFAULT_THRESHOLD
        self.texts = texts or self.DEFAULT_TEXTS
        self.model = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT
        return ModelInfo(
            model="yoloworld",
            variant=variant,
            group=ModelGroup.RED
            if variant == cls.DEFAULT_VARIANT
            else ModelGroup.GENERALITY,
            task=ModelTask.CV_OBJECT_DET,
            source=ModelSource.CUSTOM,
            framework=Framework.TORCH,
        )

    def load_model(self, dtype_override=None):
        from .src.utils import init_detector, Config, get_base_cfg
        from .src.model import MODELS

        checkpoint = str(get_file(f"test_files/pytorch/yoloworld/{self.variant}.pth"))
        base_cfg = get_base_cfg(variant=self.variant)
        self.config = Config(base_cfg)
        self.config.load_from = checkpoint
        self.model = init_detector(self.config, checkpoint)
        if dtype_override is not None:
            self.model = self.model.to(dtype_override)
        self.texts = [[t.strip()] for t in self.texts.split(",")] + [[" "]]
        self.model.reparameterize(self.texts)
        return self.model

    def load_inputs(self, dtype_override=None):

        from .src.utils import get_test_pipeline_cfg
        from .src.model import Compose

        self.image_file = get_file("https://ultralytics.com/images/bus.jpg")
        test_pipeline_cfg = get_test_pipeline_cfg(cfg=self.config)
        test_pipeline = Compose(test_pipeline_cfg)
        data_info = dict(img_id=0, img_path=self.image_file, texts=self.texts)
        data_info = test_pipeline(data_info)
        if dtype_override is not None:
            data_info["inputs"] = data_info["inputs"].to(dtype_override)
        self.data_batch = dict(
            inputs=data_info["inputs"].unsqueeze(0),
            data_samples=[data_info["data_samples"]],
        )
        self.data_batch = self.model.data_preprocessor(self.data_batch, False)
        return self.data_batch

    def post_process(self, output, output_dir):

        import supervision as sv
        import cv2
        import os
        import os.path as osp
        from .src.model import LabelAnnotator

        output = self.model.post_process(output, self.data_batch["data_samples"])
        pred_instances = output[0].pred_instances
        pred_instances = pred_instances[pred_instances.scores.float() > self.threshold]
        if len(pred_instances.scores) > self.topk:
            indices = pred_instances.scores.float().topk(self.topk)[1]
            pred_instances = pred_instances[indices]
        pred_instances = pred_instances.numpy()
        pred_instances["labels"] = pred_instances["labels"].astype(int)
        if "masks" in pred_instances:
            masks = pred_instances["masks"]
        else:
            masks = None
        detections = sv.Detections(
            xyxy=pred_instances["bboxes"],
            class_id=pred_instances["labels"],
            confidence=pred_instances["scores"],
            mask=masks,
        )
        labels = [
            f"{self.texts[class_id][0]} {confidence:0.2f}"
            for class_id, confidence in zip(detections.class_id, detections.confidence)
        ]
        image = cv2.imread(self.image_file)
        LABEL_ANNOTATOR = LabelAnnotator(
            text_padding=4, text_scale=0.5, text_thickness=1
        )
        BOUNDING_BOX_ANNOTATOR = sv.BoundingBoxAnnotator(thickness=1)
        MASK_ANNOTATOR = sv.MaskAnnotator()
        image = BOUNDING_BOX_ANNOTATOR.annotate(image, detections)
        image = LABEL_ANNOTATOR.annotate(image, detections, labels=labels)
        if not osp.exists(output_dir):
            os.mkdir(output_dir)
        cv2.imwrite(osp.join(output_dir, osp.basename(self.image_file)), image)
