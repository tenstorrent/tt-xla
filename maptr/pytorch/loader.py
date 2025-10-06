# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
MAPTR model loader implementation
"""
import torch
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
from ...tools.utils import get_file
from .src.maptr import (
    build_model,
    model_cfg,
    gkt_cfg,
    lss_cfg,
    lidar_encoder_cfg,
    fuser_cfg,
    load_checkpoint,
    ConfigDict,
)
import numpy as np
import copy


class ModelVariant(StrEnum):
    """Available MAPTR model variants."""

    TINY_R50_24E_BEVFORMER = "tiny_r50_24e_bevformer"
    TINY_R50_24E_BEVFORMER_T4 = "tiny_r50_24e_bevformer_t4"
    TINY_R50_24E = "tiny_r50_24e"
    TINY_R50_110E = "tiny_r50_110e"
    TINY_R50_24E_T4 = "tiny_r50_24e_t4"
    NANO_R18_110E = "nano_r18_110e"
    TINY_R50_24E_BEVPOOL = "tiny_r50_24e_bevpool"
    TINY_FUSION_24E = "tiny_fusion_24e"
    TINY_R50_24E_AV2 = "tiny_r50_24e_av2"


class ModelLoader(ForgeModel):
    """MAPTR model loader implementation."""

    # Dictionary of available model variants using structured configs
    _VARIANTS = {
        ModelVariant.TINY_R50_24E_BEVFORMER: ModelConfig(
            pretrained_model_name="tiny_r50_24e_bevformer",
        ),
        ModelVariant.TINY_R50_24E_BEVFORMER_T4: ModelConfig(
            pretrained_model_name="tiny_r50_24e_bevformer_t4",
        ),
        ModelVariant.TINY_R50_24E: ModelConfig(
            pretrained_model_name="tiny_r50_24e",
        ),
        ModelVariant.TINY_R50_110E: ModelConfig(
            pretrained_model_name="tiny_r50_110e",
        ),
        ModelVariant.TINY_R50_24E_T4: ModelConfig(
            pretrained_model_name="tiny_r50_24e_t4",
        ),
        ModelVariant.NANO_R18_110E: ModelConfig(
            pretrained_model_name="nano_r18_110e",
        ),
        ModelVariant.TINY_R50_24E_BEVPOOL: ModelConfig(
            pretrained_model_name="tiny_r50_24e_bevpool",
        ),
        ModelVariant.TINY_FUSION_24E: ModelConfig(
            pretrained_model_name="tiny_fusion_24e",
        ),
        ModelVariant.TINY_R50_24E_AV2: ModelConfig(
            pretrained_model_name="tiny_r50_24e_av2",
        ),
    }

    # Default variant to use
    DEFAULT_VARIANT = ModelVariant.TINY_R50_24E_BEVFORMER

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
        return ModelInfo(
            model="maptr",
            variant=variant,
            group=ModelGroup.RED,
            task=ModelTask.REALTIME_MAP_CONSTRUCTION,
            source=ModelSource.GITHUB,
            framework=Framework.TORCH,
        )

    def load_model(self):
        """Load pretrained MAPTR model for this instance's variant.

        Returns:
            torch.nn.Module: The MAPTR model instance.
        """
        # Get the variant name from the instance's variant config
        variant_name = self._variant_config.pretrained_model_name

        # Get pretrained weights (skip for AV2 variant as no weights are available)
        if variant_name != "tiny_r50_24e_av2":
            model_path = str(
                get_file(f"test_files/pytorch/maptr/maptr_{variant_name}.pth")
            )

        # Create deep copy to prevent global state contamination
        local_model_cfg = copy.deepcopy(model_cfg)

        # Update config
        if variant_name in ["tiny_r50_24e_bevformer_t4", "tiny_r50_24e_t4"]:
            local_model_cfg["pts_bbox_head"]["transformer"]["len_can_bus"] = 6

        if variant_name in [
            "tiny_r50_24e",
            "tiny_r50_110e",
            "tiny_r50_24e_t4",
            "nano_r18_110e",
            "tiny_fusion_24e",
            "tiny_r50_24e_av2",
        ]:
            local_model_cfg["pts_bbox_head"]["transformer"]["encoder"][
                "transformerlayers"
            ]["attn_cfgs"][1] = copy.deepcopy(gkt_cfg)

        if variant_name == "nano_r18_110e":

            # Define nano BEV dimensions
            nano_bev_h = 80
            nano_bev_w = 40

            local_model_cfg["img_backbone"]["depth"] = 18
            local_model_cfg["img_neck"]["in_channels"] = [512]
            local_model_cfg["pts_bbox_head"]["num_vec"] = 100
            local_model_cfg["pts_bbox_head"]["transformer"]["decoder"]["num_layers"] = 2
            local_model_cfg["pts_bbox_head"]["transformer"]["decoder"][
                "transformerlayers"
            ]["attn_cfgs"][0]["num_heads"] = 4

            # Update BEV dimensions
            local_model_cfg["pts_bbox_head"]["bev_h"] = nano_bev_h
            local_model_cfg["pts_bbox_head"]["bev_w"] = nano_bev_w
            local_model_cfg["pts_bbox_head"]["positional_encoding"][
                "row_num_embed"
            ] = nano_bev_h
            local_model_cfg["pts_bbox_head"]["positional_encoding"][
                "col_num_embed"
            ] = nano_bev_w

        if variant_name == "tiny_r50_24e_bevpool":

            # BevPool specific parameters
            bevpool_pc_range = [-15.0, -30.0, -10.0, 15.0, 30.0, 10.0]
            bevpool_voxel_size = [0.15, 0.15, 20.0]

            # Replace encoder with BevPool's LSSTransform
            local_model_cfg["pts_bbox_head"]["transformer"]["encoder"] = copy.deepcopy(
                lss_cfg
            )

            # Override parameters in encoder
            local_model_cfg["pts_bbox_head"]["transformer"]["encoder"][
                "pc_range"
            ] = bevpool_pc_range
            local_model_cfg["pts_bbox_head"]["transformer"]["encoder"][
                "voxel_size"
            ] = bevpool_voxel_size

            # Override parameters in bbox_coder
            local_model_cfg["pts_bbox_head"]["bbox_coder"][
                "pc_range"
            ] = bevpool_pc_range
            local_model_cfg["pts_bbox_head"]["bbox_coder"][
                "voxel_size"
            ] = bevpool_voxel_size

        if variant_name == "tiny_fusion_24e":
            # Fusion specific parameters
            fusion_voxel_size = [0.1, 0.1, 0.2]

            # Add fusion-specific configurations
            local_model_cfg["modality"] = "fusion"
            local_model_cfg["lidar_encoder"] = copy.deepcopy(lidar_encoder_cfg)
            local_model_cfg["pts_bbox_head"]["transformer"]["modality"] = "fusion"
            local_model_cfg["pts_bbox_head"]["transformer"]["fuser"] = copy.deepcopy(
                fuser_cfg
            )

            # Override voxel size in lidar_encoder
            local_model_cfg["lidar_encoder"]["voxelize"][
                "voxel_size"
            ] = fusion_voxel_size

            # Override parameters in bbox_coder
            local_model_cfg["pts_bbox_head"]["bbox_coder"][
                "voxel_size"
            ] = fusion_voxel_size

        if variant_name == "tiny_r50_24e_av2":
            # AV2 specific configurations
            av2_point_cloud_range = [-30.0, -15.0, -2.0, 30.0, 15.0, 2.0]
            av2_bev_h = 100
            av2_bev_w = 200
            av2_post_center_range = [-35, -20, -35, -20, 35, 20, 35, 20]

            # Update point cloud range
            local_model_cfg["pts_bbox_head"]["transformer"]["encoder"][
                "pc_range"
            ] = av2_point_cloud_range
            local_model_cfg["pts_bbox_head"]["transformer"]["encoder"][
                "transformerlayers"
            ]["attn_cfgs"][1]["pc_range"] = av2_point_cloud_range
            local_model_cfg["pts_bbox_head"]["bbox_coder"][
                "pc_range"
            ] = av2_point_cloud_range

            # Update BEV dimensions
            local_model_cfg["pts_bbox_head"]["bev_h"] = av2_bev_h
            local_model_cfg["pts_bbox_head"]["bev_w"] = av2_bev_w
            local_model_cfg["pts_bbox_head"]["positional_encoding"][
                "row_num_embed"
            ] = av2_bev_h
            local_model_cfg["pts_bbox_head"]["positional_encoding"][
                "col_num_embed"
            ] = av2_bev_w

            # Update post center range
            local_model_cfg["pts_bbox_head"]["bbox_coder"][
                "post_center_range"
            ] = av2_post_center_range

            # Set number of cameras to 7 for AV2
            local_model_cfg["pts_bbox_head"]["transformer"]["num_cams"] = 7
            local_model_cfg["pts_bbox_head"]["transformer"]["encoder"][
                "transformerlayers"
            ]["attn_cfgs"][1]["num_cams"] = 7
            local_model_cfg["pts_bbox_head"]["transformer"]["encoder"][
                "transformerlayers"
            ]["attn_cfgs"][1]["pc_range"] = av2_point_cloud_range

        # wrap the model config
        self.cfg = ConfigDict(local_model_cfg)

        # Build model
        model = build_model(self.cfg)

        # Load pretrained weights (skip for AV2 variant as no weights are available)
        if variant_name != "tiny_r50_24e_av2":
            load_checkpoint(model, model_path, map_location="cpu")
        model.eval()

        return model

    def load_inputs(self):
        """Prepare dummy sample inputs for the MAPTR model.

        This function creates synthetic test data mimicking the structure expected by
        MAPTR for different datasets. It does not load real data, but instead
        generates placeholder inputs for testing purposes:

        - img_shape: List of image shapes, one per camera view.
        - lidar2img: List of 4x4 projection matrices (identity matrices here as placeholders).
        - can_bus: Vehicle state vector (zeros used as dummy values).
        - img: Random image tensor simulating camera views (6 for NuScenes, 7 for AV2).
        - (bevpool variant only) camera2ego, camera_intrinsics, img_aug_matrix, lidar2ego:
          Identity matrices used as placeholders.

        Returns:
            dict: Dictionary containing:
                - ``img_metas`` (list): Metadata with image shapes, projection matrices,
                CAN bus vector, and a dummy scene token (plus bevpool-specific fields when
                using the tiny_r50_24e_bevpool variant).
                - ``img`` (list): Randomized image tensor of shape (1, num_cams, 3, 480, 800)
                  where num_cams is 6 for NuScenes variants and 7 for AV2 variants.
        """

        # Determine number of cameras based on variant
        num_cams = (
            7 if self._variant_config.pretrained_model_name == "tiny_r50_24e_av2" else 6
        )

        img_shape = [(480, 800, 3)] * num_cams
        lidar2img = [np.eye(4) for _ in range(num_cams)]
        can_bus = np.zeros(18, dtype=np.float64)
        img = torch.randn(1, num_cams, 3, 480, 800)

        data = {
            "img_metas": [
                [
                    {
                        "img_shape": img_shape,
                        "lidar2img": lidar2img,
                        "can_bus": can_bus,
                        "scene_token": "fcbccedd61424f1b85dcbf8f897f9754",
                    }
                ]
            ],
            "img": [img],
        }

        if self._variant_config.pretrained_model_name == "tiny_r50_24e_bevpool":
            data["img_metas"][0][0].update(
                dict(
                    camera2ego=[np.eye(4, dtype=np.float32) for _ in range(6)],
                    camera_intrinsics=[np.eye(4, dtype=np.float32) for _ in range(6)],
                    img_aug_matrix=[np.eye(4, dtype=np.float64) for _ in range(6)],
                    lidar2ego=np.eye(4, dtype=np.float32),
                )
            )

        if self._variant_config.pretrained_model_name == "tiny_fusion_24e":
            data["points"] = [[torch.randn(196628, 5)]]

        return data
