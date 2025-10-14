# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
SSR model loader implementation (aligned with BEVFormer loader style)
"""
import torch
import numpy as np
from typing import Optional
from ...base import ForgeModel
from ...config import ModelGroup, ModelTask, ModelSource, Framework, StrEnum, ModelInfo
from .src.model import (
    SSR,
    img_backbone,
    img_neck,
    pts_bbox_head,
    latent_world_model,
)
from third_party.tt_forge_models.bevdepth.pytorch.src.model import load_checkpoint
from ...tools.utils import get_file


class ModelVariant(StrEnum):
    SSR_E2E = "SSR_e2e"


class ModelLoader(ForgeModel):
    """SSR model loader implementation for autonomous driving tasks."""

    DEFAULT_VARIANT = ModelVariant.SSR_E2E

    def __init__(self, variant=None):
        super().__init__(variant)
        self.processor = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        return ModelInfo(
            model="ssr",
            variant=variant,
            group=ModelGroup.RED,
            task=ModelTask.CV_OBJECT_DET,
            source=ModelSource.CUSTOM,
            framework=Framework.TORCH,
        )

    def load_model(self, **kwargs):
        """Load and return the SSR model instance with default settings."""
        model = SSR(
            pts_bbox_head=pts_bbox_head,
            img_neck=img_neck,
            img_backbone=img_backbone,
            latent_world_model=latent_world_model,
            use_grid_mask=True,
            video_test_mode=True,
        )
        checkpoint_path = str(get_file("test_files/pytorch/ssr/ssr.pth"))
        load_checkpoint(model, checkpoint_path)

        model.eval()
        return model

    def load_inputs(self, **kwargs):
        """Return sample inputs for the SSR model with default settings."""
        torch.manual_seed(42)
        np.random.seed(42)

        img_norm_cfg = {
            "mean": np.array([123.675, 116.28, 103.53], dtype=np.float32),
            "std": np.array([58.395, 57.12, 57.375], dtype=np.float32),
            "to_rgb": True,
        }
        img_shapes = [(384, 640, 3)] * 6
        pad_shapes = [(384, 640, 3)] * 6
        ori_shapes = [(360, 640, 3)] * 6

        img = torch.randn(1, 6, 3, 384, 640)
        points = torch.rand(34752, 5)

        lidar2img = [np.random.rand(4, 4)]
        can_bus = np.random.rand(
            18,
        )

        # Minimal placeholders for planning inputs
        ego_his_trajs = torch.tensor([[[[0.0757, 4.2529], [0.0757, 4.2529]]]])
        ego_fut_trajs = torch.tensor(
            [
                [
                    [
                        [0.0757, 4.2529],
                        [0.2000, 4.2128],
                        [0.2907, 4.1843],
                        [0.3986, 4.1790],
                        [0.5144, 4.2608],
                        [0.6900, 4.2691],
                    ]
                ]
            ]
        )
        ego_fut_masks = torch.tensor([[[[1.0, 1.0, 1.0, 1.0, 1.0, 1.0]]]])
        ego_fut_cmd = torch.tensor([[[[1.0, 0.0, 0.0]]]])
        ego_lcf_feat = torch.tensor(
            [
                [
                    [
                        [
                            -3.4331e-02,
                            8.5224e00,
                            -6.0694e-01,
                            -7.6344e-02,
                            -2.7797e-03,
                            4.0840e00,
                            1.8500e00,
                            8.5641e00,
                            -1.2268e-01,
                        ]
                    ]
                ]
            ]
        )
        map_gt_labels_3d = torch.tensor([0, 0, 0, 1, 2, 2, 2])
        map_gt_bboxes_3d = torch.randn(7, 20, 2)

        class LiDARInstance3DBoxes:
            def __init__(self, boxes: torch.Tensor):
                """
                Custom wrapper for LiDAR 3D boxes.
                """
                if not isinstance(boxes, torch.Tensor):
                    raise TypeError("Input must be a torch.Tensor")
                if boxes.ndim != 2 or boxes.shape[1] != 9:
                    raise ValueError("Input tensor must have shape (N, 9)")

                self.tensor = boxes
                self.shape = boxes.shape

            def __getitem__(self, index):
                return self.tensor[index]

        boxes = torch.rand((11, 9), dtype=torch.float32)
        gt_bboxes_3d = LiDARInstance3DBoxes(boxes)
        gt_labels_3d = torch.tensor([8, 8, 8, 8, 8, 8, 0, 8, 8, 0, 8])
        gt_attr_labels = torch.randn(11, 34)

        img_metas = {
            "ori_shape": ori_shapes,
            "img_shape": img_shapes,
            "lidar2img": lidar2img,
            "pad_shape": pad_shapes,
            "scale_factor": 1.0,
            "flip": False,
            "pcd_horizontal_flip": False,
            "pcd_vertical_flip": False,
            "img_norm_cfg": img_norm_cfg,
            "sample_idx": "3e8750f331d7499e9b5123e9eb70f2e2",
            "prev_idx": "",
            "next_idx": "3950bd41f74548429c0f7700ff3d8269",
            "pcd_scale_factor": 1.0,
            "scene_token": "fcbccedd61424f1b85dcbf8f897f9754",
            "can_bus": can_bus,
        }

        input_dict = {
            "rescale": True,
            "img_metas": [[img_metas]],
            "points": [[points]],
            "gt_bboxes_3d": [[gt_bboxes_3d]],
            "gt_labels_3d": [[gt_labels_3d]],
            "img": [img],
            "ego_his_trajs": [ego_his_trajs],
            "ego_fut_trajs": [ego_fut_trajs],
            "ego_fut_masks": [ego_fut_masks],
            "ego_fut_cmd": [ego_fut_cmd],
            "ego_lcf_feat": [ego_lcf_feat],
            "gt_attr_labels": [[gt_attr_labels]],
            "map_gt_labels_3d": [map_gt_labels_3d],
            "map_gt_bboxes_3d": [map_gt_bboxes_3d],
        }

        return input_dict
