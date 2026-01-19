# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
BEVFormer model loader implementation
"""
import torch
from typing import Optional
from ...base import ForgeModel
from ...config import (
    ModelGroup,
    ModelTask,
    ModelSource,
    Framework,
    StrEnum,
    ModelInfo,
    ModelConfig,
)

from .src.model import (
    BEVFormer,
    BEVFormerV2,
    get_bevformer_model,
    get_bevformer_v2_model,
    load_checkpoint_bev,
)
from .src.model_utils import build_inputs
from ...tools.utils import get_file


class ModelVariant(StrEnum):
    """Available BEVFormer model variants."""

    BEVFORMER_TINY = "BEVFormer-tiny"
    BEVFORMER_SMALL = "BEVFormer-small"
    BEVFORMER_BASE = "BEVFormer-base"
    BEVFORMER_V2_R50_T1_BASE = "bevformerv2-r50-t1-base"
    BEVFORMER_V2_R50_T1 = "bevformerv2-r50-t1"
    BEVFORMER_V2_R50_T2 = "bevformerv2-r50-t2"
    BEVFORMER_V2_R50_T8 = "bevformerv2-r50-t8"


class ModelLoader(ForgeModel):
    """BEVFormer model loader implementation for autonomous driving tasks."""

    _VARIANTS = {
        ModelVariant.BEVFORMER_TINY: ModelConfig(
            pretrained_model_name="BEVFormer-tiny"
        ),
        ModelVariant.BEVFORMER_SMALL: ModelConfig(
            pretrained_model_name="BEVFormer-small"
        ),
        ModelVariant.BEVFORMER_BASE: ModelConfig(
            pretrained_model_name="BEVFormer-base"
        ),
        ModelVariant.BEVFORMER_V2_R50_T1_BASE: ModelConfig(
            pretrained_model_name="bevformerv2-r50-t1-base"
        ),
        ModelVariant.BEVFORMER_V2_R50_T1: ModelConfig(
            pretrained_model_name="bevformerv2-r50-t1"
        ),
        ModelVariant.BEVFORMER_V2_R50_T2: ModelConfig(
            pretrained_model_name="bevformerv2-r50-t2"
        ),
        ModelVariant.BEVFORMER_V2_R50_T8: ModelConfig(
            pretrained_model_name="bevformerv2-r50-t8"
        ),
    }
    # Default variant to use
    DEFAULT_VARIANT = ModelVariant.BEVFORMER_TINY

    def __init__(self, variant=None):
        """Initialize ModelLoader with specified variant.
        Args:
            variant: Optional ModelVariant specifying which variant to use.
                     If None, DEFAULT_VARIANT is used.
        """
        super().__init__(variant)
        # Configuration parameters
        self.processor = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        """Implementation method for getting model info with validated variant.
        Args:
            variant: Optional ModelVariant specifying which variant to use.
                     If None, DEFAULT_VARIANT is used.
        Returns:
            ModelInfo: Information about the model and variant
        """

        return ModelInfo(
            model="bevformer",
            variant=variant,
            group=ModelGroup.RED
            if variant == ModelVariant.BEVFORMER_TINY
            else ModelGroup.GENERALITY,
            task=ModelTask.CV_OBJECT_DET,
            source=ModelSource.CUSTOM,
            framework=Framework.TORCH,
        )

    def load_model(self, variant: Optional["ModelVariant"] = None, **kwargs):
        """Load and return the BEVFormer model instance with default settings.
        Returns:
            Torch model: The BEVFormer model instance.
        """
        variant_str = str(self._variant) if self._variant else str(self.DEFAULT_VARIANT)
        if (
            variant_str == ModelVariant.BEVFORMER_V2_R50_T1_BASE.value
            or variant_str == ModelVariant.BEVFORMER_V2_R50_T1.value
            or variant_str == ModelVariant.BEVFORMER_V2_R50_T2.value
            or variant_str == ModelVariant.BEVFORMER_V2_R50_T8.value
        ):
            (
                img_backbone,
                pts_bbox_head,
                img_neck,
                fcos3d_bbox_head,
                frames,
            ) = get_bevformer_v2_model(variant_str)
            model = BEVFormerV2(
                img_backbone=img_backbone,
                pts_bbox_head=pts_bbox_head,
                img_neck=img_neck,
                fcos3d_bbox_head=fcos3d_bbox_head,
                frames=frames,
                use_grid_mask=True,
                video_test_mode=False,
                num_levels=4,
                num_mono_levels=5,
            )
        else:
            img_backbone, pts_bbox_head, img_neck = get_bevformer_model(variant_str)
            model = BEVFormer(
                img_backbone=img_backbone,
                pts_bbox_head=pts_bbox_head,
                img_neck=img_neck,
                use_grid_mask=True,
                video_test_mode=True,
            )
        if variant_str == ModelVariant.BEVFORMER_SMALL.value:
            checkpoint_path = str(
                get_file("test_files/pytorch/bevformer/bevformer_small_epoch_24.pth")
            )
        elif variant_str == ModelVariant.BEVFORMER_BASE.value:
            checkpoint_path = str(
                get_file("test_files/pytorch/bevformer/bevformer_r101_dcn_24ep.pth")
            )
        elif variant_str == ModelVariant.BEVFORMER_V2_R50_T1_BASE.value:
            checkpoint_path = str(
                get_file(
                    "test_files/pytorch/bevformer/bevformerv2_t1_base_epoch_24.pth"
                )
            )
        elif variant_str == ModelVariant.BEVFORMER_V2_R50_T1.value:
            checkpoint_path = str(
                get_file("test_files/pytorch/bevformer/bevformerv2_r50_t1_epoch_24.pth")
            )
        elif variant_str == ModelVariant.BEVFORMER_V2_R50_T2.value:
            checkpoint_path = str(
                get_file("test_files/pytorch/bevformer/bevformerv2_r50_t2_epoch_24.pth")
            )
        elif variant_str == ModelVariant.BEVFORMER_V2_R50_T8.value:
            checkpoint_path = str(
                get_file("test_files/pytorch/bevformer/bevformerv2_r50_t8_epoch_24.pth")
            )
        else:
            checkpoint_path = str(
                get_file("test_files/pytorch/bevformer/bevformer_tiny_epoch_24.pth")
            )
        checkpoint = load_checkpoint_bev(
            model,
            checkpoint_path,
            map_location="cpu",
        )
        model.eval()
        return model

    def load_inputs(self, variant: Optional["ModelVariant"] = None, **kwargs):
        """Return sample inputs for the BEVFormer model with default settings.
        Returns:
            dict: A dictionary of input tensors and metadata suitable for the model.
        """

        inputs_dict = build_inputs()
        variant_str = str(self._variant) if self._variant else str(self.DEFAULT_VARIANT)
        if (
            variant_str == ModelVariant.BEVFORMER_V2_R50_T1_BASE.value
            or variant_str == ModelVariant.BEVFORMER_V2_R50_T1.value
            or variant_str == ModelVariant.BEVFORMER_V2_R50_T2.value
            or variant_str == ModelVariant.BEVFORMER_V2_R50_T8.value
        ):
            img = torch.randn(1, 6, 3, 640, 1600)
            img_metas = {
                "img_shape": inputs_dict["img_shapes"],
                "lidar2img": inputs_dict["lidar2img"],
                "lidar2cam": inputs_dict["lidar2cam"],
                "pad_shape": inputs_dict["pad_shape"],
                "scale_factor": 1.0,
                "flip": False,
                "pcd_horizontal_flip": False,
                "pcd_vertical_flip": False,
                "img_norm_cfg": inputs_dict["img_norm_cfg"],
                "sample_idx": "3e8750f331d7499e9b5123e9eb70f2e2",
                "prev_idx": "",
                "next_idx": "3950bd41f74548429c0f7700ff3d8269",
                "pcd_scale_factor": 1.0,
                "scene_token": "fcbccedd61424f1b85dcbf8f897f9754",
            }
            input_dict = {
                "rescale": True,
                "img_metas": [[{0: img_metas}]],
                "img": [img],
                "ego2global_translation": inputs_dict["ego2global_translation"],
                "ego2global_rotation": inputs_dict["ego2global_rotation"],
                "lidar2ego_translation": inputs_dict["lidar2ego_translation"],
                "lidar2ego_rotation": inputs_dict["lidar2ego_rotation"],
            }
        else:
            img = torch.randn(1, 6, 3, 480, 800)
            img_metas = {
                "img_shape": inputs_dict["img_shapes"],
                "lidar2img": inputs_dict["lidar2img"],
                "scale_factor": 1.0,
                "flip": False,
                "pcd_horizontal_flip": False,
                "pcd_vertical_flip": False,
                "sample_idx": "3e8750f331d7499e9b5123e9eb70f2e2",
                "prev_idx": "",
                "next_idx": "3950bd41f74548429c0f7700ff3d8269",
                "pcd_scale_factor": 1.0,
                "scene_token": "fcbccedd61424f1b85dcbf8f897f9754",
                "can_bus": inputs_dict["can_bus"],
            }
            input_dict = {"rescale": True, "img_metas": [[img_metas]], "img": [img]}

        return input_dict
