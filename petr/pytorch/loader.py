# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
PETR model loader implementation
"""
import torch
from typing import Optional
import copy
import numpy as np

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
from .src.petr import (
    build_model,
    load_checkpoint,
    ConfigDict,
    model_cfg,
    resnet50_dcn_backbone_cfg,
    LiDARInstance3DBoxes,
)


class ModelVariant(StrEnum):
    """Available PETR model variants."""

    VOVNET_GRIDMASK_P4_800X320 = "vovnet_gridmask_p4_800x320"
    VOVNET_GRIDMASK_P4_1600X640 = "vovnet_gridmask_p4_1600x640"
    R50DCN_GRIDMASK_C5 = "r50dcn_gridmask_c5"
    R50DCN_GRIDMASK_P4 = "r50dcn_gridmask_p4"


class ModelLoader(ForgeModel):
    """PETR model loader implementation."""

    # Dictionary of available model variants using structured configs
    _VARIANTS = {
        ModelVariant.VOVNET_GRIDMASK_P4_800X320: ModelConfig(
            pretrained_model_name="vovnet_gridmask_p4_800x320",
        ),
        ModelVariant.VOVNET_GRIDMASK_P4_1600X640: ModelConfig(
            pretrained_model_name="vovnet_gridmask_p4_1600x640",
        ),
        ModelVariant.R50DCN_GRIDMASK_C5: ModelConfig(
            pretrained_model_name="r50dcn_gridmask_c5",
        ),
        ModelVariant.R50DCN_GRIDMASK_P4: ModelConfig(
            pretrained_model_name="r50dcn_gridmask_p4",
        ),
    }

    # Default variant to use
    DEFAULT_VARIANT = ModelVariant.VOVNET_GRIDMASK_P4_800X320

    def __init__(self, variant: Optional[ModelVariant] = None):
        """Initialize ModelLoader with specified variant.

        Args:
            variant: Optional ModelVariant specifying which variant to use.
                     If None, DEFAULT_VARIANT is used.
        """
        super().__init__(variant)

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        """Get model information for dashboard and metrics reporting.

        Args:
            variant: Optional ModelVariant specifying which variant to use.
                     If None, DEFAULT_VARIANT is used.

        Returns:
            ModelInfo: Information about the model and variant.
        """
        if variant is None:
            variant = cls.DEFAULT_VARIANT

        if variant == ModelVariant.VOVNET_GRIDMASK_P4_800X320:
            group = ModelGroup.RED
        else:
            group = ModelGroup.GENERALITY

        return ModelInfo(
            model="petr",
            variant=variant,
            group=group,
            task=ModelTask.MULTIVIEW_3D_OBJECT_DET,
            source=ModelSource.GITHUB,
            framework=Framework.TORCH,
        )

    def load_model(self):
        """Load pretrained PETR model for this instance's variant.

        Returns:
            torch.nn.Module: The PETR model instance.
        """
        # Get the variant name from the instance's variant config
        variant_name = self._variant_config.pretrained_model_name

        # Get pretrained weights
        model_path = str(get_file(f"test_files/pytorch/petr/{variant_name}.pth"))

        # Create deep copy to prevent global state contamination
        local_model_cfg = copy.deepcopy(model_cfg)

        # Modify backbone and neck based on variant
        if variant_name == "r50dcn_gridmask_c5":
            # ResNet50 DCN with C5 (no neck)
            local_model_cfg["img_backbone"] = copy.deepcopy(resnet50_dcn_backbone_cfg)
            local_model_cfg["img_backbone"]["out_indices"] = (3,)
            del local_model_cfg["img_neck"]
            local_model_cfg["pts_bbox_head"]["in_channels"] = 2048

        elif variant_name == "r50dcn_gridmask_p4":
            # ResNet50 DCN with P4 (with neck)
            local_model_cfg["img_backbone"] = copy.deepcopy(resnet50_dcn_backbone_cfg)
            local_model_cfg["img_neck"]["in_channels"] = [1024, 2048]

        # Add checkpoint flag for all variants except base 800x320
        if variant_name != "vovnet_gridmask_p4_800x320":
            local_model_cfg["pts_bbox_head"]["transformer"]["decoder"][
                "transformerlayers"
            ]["with_cp"] = True

        # Wrap the model config
        self.cfg = ConfigDict(local_model_cfg)

        # Build model
        model = build_model(self.cfg)

        # Load pretrained weights
        load_checkpoint(model, model_path, map_location="cpu")
        model.eval()

        return model

    def load_inputs(self):
        """Prepare dummy sample inputs for the PETR model.

        This function creates synthetic test data mimicking the structure expected by
        PETR for NuScenes dataset. It does not load real data, but instead
        generates placeholder inputs for testing purposes.

        The input dimensions are automatically selected based on the model variant:
        - vovnet_gridmask_p4_800x320: 320x800 images
        - vovnet_gridmask_p4_1600x640: 640x1600 images
        - r50dcn_gridmask_c5 & r50dcn_gridmask_p4: 512x1408 images

        Returns:
            dict: Dictionary containing:
                - ``img_metas`` (list): Nested list containing a dictionary with metadata:
                    - ``img_shape`` (list): Original image dimensions [(H, W, 3)] * 6 cameras
                    - ``pad_shape`` (list): Padded image dimensions [(H, W, 3)] * 6 cameras
                    - ``lidar2img`` (list): 4x4 projection matrices (identity matrices) for 6 cameras
                    - ``box_type_3d`` (class): LiDARInstance3DBoxes class for wrapping 3D bounding boxes
                - ``img`` (list): Random image tensor of shape [1, 6, 3, H, W] where 6 is the number of cameras
        """
        variant_name = self._variant_config.pretrained_model_name

        # Determine image dimensions based on variant
        if variant_name == "vovnet_gridmask_p4_800x320":
            img_h, img_w = 320, 800
        elif variant_name == "vovnet_gridmask_p4_1600x640":
            img_h, img_w = 640, 1600
        elif variant_name in ["r50dcn_gridmask_c5", "r50dcn_gridmask_p4"]:
            img_h, img_w = 512, 1408
        else:
            raise ValueError(f"Unknown variant: {variant_name}")

        # Number of cameras (6 for NuScenes)
        num_cams = 6

        # Create dummy inputs
        img_shape = [(img_h, img_w, 3)] * num_cams
        pad_shape = [(img_h, img_w, 3)] * num_cams
        lidar2img = [np.eye(4) for _ in range(num_cams)]
        img = torch.randn(1, num_cams, 3, img_h, img_w)

        data = {
            "img_metas": [
                [
                    {
                        "img_shape": img_shape,
                        "pad_shape": pad_shape,
                        "lidar2img": lidar2img,
                        "box_type_3d": LiDARInstance3DBoxes,
                    }
                ]
            ],
            "img": [img],
        }

        return data
