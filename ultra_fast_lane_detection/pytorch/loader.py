# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Ultra-Fast-Lane-Detection model loader for tt-forge-models
"""

import torch
import os
import cv2
from typing import Optional, List, Dict, Any
from dataclasses import dataclass
from PIL import Image
import numpy as np
import torchvision.transforms as transforms

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
from ...tools.utils import get_file

from .src.utils import (
    load_lane_detection_model,
    preprocess_image,
    postprocess_detections,
    visualize_lanes,
)

from .src.utils import tusimple_row_anchor as TUSIMPLE_ROW_ANCHOR
from .src.utils import culane_row_anchor as CULANE_ROW_ANCHOR


@dataclass
class LaneDetectionConfig(ModelConfig):
    """Configuration specific to Ultra-Fast-Lane-Detection models"""

    dataset: str  # 'CULane' or 'Tusimple'
    backbone: str  # ResNet backbone variant (e.g., '18', '34', '50')
    griding_num: int  # Number of grid cells
    cls_num_per_lane: int  # Number of classification points per lane
    img_w: int  # Output image width
    img_h: int  # Output image height
    row_anchor: List[int]  # Row anchor points for lane detection
    input_size: tuple  # Model input size (height, width)


class ModelVariant(StrEnum):
    """Available Ultra-Fast-Lane-Detection model variants."""

    # TuSimple variants (Highway/Freeway focused)
    TUSIMPLE_RESNET18 = "tusimple_resnet18"
    TUSIMPLE_RESNET34 = "tusimple_resnet34"

    # CULane variants (Urban roads focused)
    CULANE_RESNET18 = "culane_resnet18"


class ModelLoader(ForgeModel):
    """Ultra-Fast-Lane-Detection model loader implementation."""

    _VARIANTS = {
        # TuSimple variants
        ModelVariant.TUSIMPLE_RESNET18: LaneDetectionConfig(
            pretrained_model_name="tusimple_18",
            dataset="Tusimple",
            backbone="18",
            griding_num=100,
            cls_num_per_lane=56,
            img_w=1280,
            img_h=720,
            row_anchor=TUSIMPLE_ROW_ANCHOR,
            input_size=(288, 800),
        ),
        ModelVariant.TUSIMPLE_RESNET34: LaneDetectionConfig(
            pretrained_model_name="tusimple_34",
            dataset="Tusimple34",
            backbone="34",
            griding_num=100,
            cls_num_per_lane=56,
            img_w=1280,
            img_h=720,
            row_anchor=TUSIMPLE_ROW_ANCHOR,
            input_size=(320, 800),
        ),
        # CULane variants
        ModelVariant.CULANE_RESNET18: LaneDetectionConfig(
            pretrained_model_name="culane_18",
            dataset="CULane",
            backbone="18",
            griding_num=200,
            cls_num_per_lane=18,
            img_w=1640,
            img_h=590,
            row_anchor=CULANE_ROW_ANCHOR,
            input_size=(288, 800),
        ),
    }

    DEFAULT_VARIANT = ModelVariant.TUSIMPLE_RESNET18

    # Model paths in test_files based on dataset type
    MODEL_PATHS = {
        "Tusimple": "test_files/pytorch/Ultrafast_lane_detection/tusimple_18.pth",
        "Tusimple34": "test_files/pytorch/Ultrafast_lane_detection/tusimple_res34.pth",
        "CULane": "test_files/pytorch/Ultrafast_lane_detection/culane_18.pth",
    }

    # Sample image paths in test_images
    SAMPLE_IMAGE_PATHS = {
        "Tusimple": "test_images/tu_simple_image.jpg",
        "Tusimple34": "test_images/tu_simple_image.jpg",
        "CULane": "test_images/culane_image.jpg",
    }

    def __init__(
        self,
        variant: Optional[ModelVariant] = None,
    ):
        """
        Initialize the Ultra-Fast-Lane-Detection model loader.

        Args:
            variant: Optional ModelVariant specifying which variant to use.
                     If None, DEFAULT_VARIANT is used.
        """
        super().__init__(variant)
        self.config: LaneDetectionConfig = self._variant_config
        self.model = None

        # Automatically determine model path based on variant's dataset using get_file
        model_path_key = self.MODEL_PATHS.get(self.config.dataset)
        if model_path_key:
            self.model_path = str(get_file(model_path_key))
        else:
            raise ValueError(
                f"No model path configured for dataset: {self.config.dataset}"
            )

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        """Get model information for dashboard and metrics reporting.

        Args:
            variant: Optional ModelVariant specifying which variant to use.
                     If None, DEFAULT_VARIANT is used.

        Returns:
            ModelInfo: Information about the model and variant
        """
        if variant is None:
            variant = cls.DEFAULT_VARIANT

        return ModelInfo(
            model="ultra-fast-lane-detection",
            variant=variant,
            group=ModelGroup.RED
            if variant
            in [ModelVariant.TUSIMPLE_RESNET18, ModelVariant.TUSIMPLE_RESNET34]
            else ModelGroup.GENERALITY,
            source=ModelSource.GITHUB,
            task=ModelTask.CV_IMAGE_SEG,
            framework=Framework.TORCH,
        )

    def load_model(self) -> torch.nn.Module:
        """
        Load the Ultra-Fast-Lane-Detection model.

        Returns:
            torch.nn.Module: Loaded model
        """
        if self.model is None:
            if not os.path.exists(self.model_path):
                raise FileNotFoundError(
                    f"Model file not found: {self.model_path}\n"
                    f"For {self.config.dataset} dataset, expected file: {self.MODEL_PATHS.get(self.config.dataset)}"
                )

            self.model = load_lane_detection_model(
                backbone=self.config.backbone,
                griding_num=self.config.griding_num,
                cls_num_per_lane=self.config.cls_num_per_lane,
                model_path=self.model_path,
                input_size=self.config.input_size,
            )
            self.model.eval()

        return self.model

    def _get_sample_image_path(self) -> Optional[str]:
        """Get sample image path for the dataset using get_file."""
        dataset = self.config.dataset
        sample_image_key = self.SAMPLE_IMAGE_PATHS.get(dataset)

        if not sample_image_key:
            return None

        try:
            sample_image_path = str(get_file(sample_image_key))
            if os.path.exists(sample_image_path):
                return sample_image_path
        except Exception as e:
            print(f"Warning: Could not get sample image: {e}")

        return None

    def load_inputs(self) -> torch.Tensor:
        """
        Load and preprocess the predefined sample image for the dataset.
        Automatically selects the correct image based on variant:
        - TuSimple variants → test_images/tu_simple_image.jpg
        - CULane variants → test_images/culane_image.jpg

        Returns:
            torch.Tensor: Preprocessed image tensor [1, 3, 288, 800]
        """
        # Get sample image from test_images based on dataset (which is determined by variant)
        dataset = self.config.dataset
        sample_image_path = self._get_sample_image_path()

        if sample_image_path and os.path.exists(sample_image_path):
            print(
                f"Using predefined sample image for {dataset} dataset: {sample_image_path}"
            )
            image = cv2.imread(sample_image_path)
            input_tensor = preprocess_image(image, input_size=self.config.input_size)
            return input_tensor
        else:
            expected_image = self.SAMPLE_IMAGE_PATHS.get(dataset)
            raise FileNotFoundError(
                f"Sample image not found for {dataset} dataset. "
                f"Expected: {expected_image}"
            )

    def post_process(
        self,
        output: torch.Tensor,
        original_image: Optional[np.ndarray] = None,
    ) -> Dict[str, Any]:
        """
        Post-process model output to extract lane detections.

        Args:
            output: Model output tensor [1, griding_num+1, cls_num_per_lane, num_lanes]
            original_image: Original input image for visualization

        Returns:
            Dictionary containing:
                - lanes: List of detected lanes, each as list of (x, y) points
                - num_lanes: Number of detected lanes
                - visualization: Annotated image (if original_image provided)
        """
        # Post-process using local utilities
        lanes, num_lanes = postprocess_detections(
            model_output=output,
            griding_num=self.config.griding_num,
            cls_num_per_lane=self.config.cls_num_per_lane,
            img_w=self.config.img_w,
            img_h=self.config.img_h,
            row_anchor=self.config.row_anchor,
        )

        result = {
            "lanes": lanes,
            "num_lanes": num_lanes,
            "dataset": self.config.dataset,
            "backbone": f"ResNet-{self.config.backbone}",
        }

        if original_image is not None:
            vis_image = visualize_lanes(
                image=original_image,
                lanes=lanes,
                img_w=self.config.img_w,
                img_h=self.config.img_h,
            )
            result["visualization"] = vis_image

        return result
